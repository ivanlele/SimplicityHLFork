use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use either::Either;
use miniscript::iter::{Tree, TreeLike};

use crate::debug::{CallTracker, DebugSymbols, TrackedCallName};
use crate::error::{Error, RichError, Span, WithSpan};
use crate::jet::JetHL;
use crate::num::{NonZeroPow2Usize, Pow2Usize};
use crate::parse::MatchPattern;
use crate::pattern::Pattern;
use crate::str::{AliasName, FunctionName, Identifier, ModuleName, WitnessName};
use crate::types::{
    AliasedType, ResolvedType, StructuralType, TypeConstructible, TypeDeconstructible, UIntType,
};
use crate::value::{UIntValue, Value};
use crate::witness::{Parameters, WitnessTypes, WitnessValues};
use crate::{impl_eq_hash, parse};

/// A program consists of the main function.
///
/// Other items such as custom functions or type aliases
/// are resolved during the creation of the AST.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program<J: JetHL> {
    main: Expression<J>,
    parameters: Parameters,
    witness_types: WitnessTypes,
    call_tracker: Arc<CallTracker>,
}

impl<J: JetHL> Program<J> {
    /// Access the main function.
    ///
    /// There is exactly one main function for each program.
    pub fn main(&self) -> &Expression<J> {
        &self.main
    }

    /// Access the parameters of the program.
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Access the witness types of the program.
    pub fn witness_types(&self) -> &WitnessTypes {
        &self.witness_types
    }

    /// Access the debug symbols of the program.
    pub fn debug_symbols(&self, file: &str) -> DebugSymbols {
        self.call_tracker.with_file(file)
    }

    /// Access the tracker of function calls.
    pub(crate) fn call_tracker(&self) -> &Arc<CallTracker> {
        &self.call_tracker
    }
}

/// An item is a component of a program.
///
/// All items except for the main function are resolved during the creation of the AST.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Item<J: JetHL> {
    /// A type alias.
    ///
    /// A stub because the alias was resolved during the creation of the AST.
    TypeAlias,
    /// A function.
    Function(Function<J>),
    /// A module, which is ignored.
    Module,
}

/// Definition of a function.
///
/// All functions except for the main function are resolved during the creation of the AST.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Function<J: JetHL> {
    /// A custom function.
    ///
    /// A stub because the definition of the function was moved to its calls in the main function.
    Custom,
    /// The main function.
    ///
    /// An expression that takes no inputs (unit) and that produces no output (unit).
    /// The expression may panic midway through, signalling failure.
    /// Otherwise, the expression signals success.
    ///
    /// This expression is evaluated when the program is run.
    Main(Expression<J>),
}

/// A statement is a component of a block expression.
///
/// Statements can define variables or run validating expressions,
/// but they never return values.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Statement<J: JetHL> {
    /// Variable assignment.
    Assignment(Assignment<J>),
    /// Expression that returns nothing (the unit value).
    Expression(Expression<J>),
}

/// Assignment of a value to a variable identifier.
#[derive(Clone, Debug)]
pub struct Assignment<J: JetHL> {
    pattern: Pattern,
    expression: Expression<J>,
    span: Span,
}

impl<J: JetHL> Assignment<J> {
    /// Access the pattern of the assignment.
    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    /// Access the expression of the assignment.
    pub fn expression(&self) -> &Expression<J> {
        &self.expression
    }

    /// Access the span of the assignment.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl_eq_hash!(Assignment<J: JetHL>; pattern, expression);

/// An expression returns a value.
#[derive(Clone, Debug)]
pub struct Expression<J: JetHL> {
    inner: ExpressionInner<J>,
    ty: ResolvedType,
    span: Span,
}

impl_eq_hash!(Expression<J: JetHL>; inner, ty);

impl<J: JetHL> Expression<J> {
    /// Access the inner expression.
    pub fn inner(&self) -> &ExpressionInner<J> {
        &self.inner
    }

    /// Access the type of the expression.
    pub fn ty(&self) -> &ResolvedType {
        &self.ty
    }

    /// Access the span of the expression.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

/// Variant of an expression.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ExpressionInner<J: JetHL> {
    /// A single expression directly returns a value.
    Single(SingleExpression<J>),
    /// A block expression first executes a series of statements inside a local scope.
    /// Then, the block returns the value of its final expression.
    /// The block returns nothing (unit) if there is no final expression.
    Block(Arc<[Statement<J>]>, Option<Arc<Expression<J>>>),
}

/// A single expression directly returns its value.
#[derive(Clone, Debug)]
pub struct SingleExpression<J: JetHL> {
    inner: SingleExpressionInner<J>,
    ty: ResolvedType,
    span: Span,
}

impl<J: JetHL> SingleExpression<J> {
    /// Create a tuple expression from the given arguments and span.
    pub fn tuple(args: Arc<[Expression<J>]>, span: Span) -> Self {
        let ty = ResolvedType::tuple(
            args.iter()
                .map(Expression::ty)
                .cloned()
                .collect::<Vec<ResolvedType>>(),
        );
        let inner = SingleExpressionInner::Tuple(args);
        Self { inner, ty, span }
    }

    /// Access the inner expression.
    pub fn inner(&self) -> &SingleExpressionInner<J> {
        &self.inner
    }

    /// Access the type of the expression.
    pub fn ty(&self) -> &ResolvedType {
        &self.ty
    }

    /// Access the span of the expression.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl_eq_hash!(SingleExpression<J: JetHL>; inner, ty);

/// Variant of a single expression.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum SingleExpressionInner<J: JetHL> {
    /// Constant value.
    Constant(Value<J>),
    /// Witness value.
    Witness(WitnessName),
    /// Parameter value.
    Parameter(WitnessName),
    /// Variable that has been assigned a value.
    Variable(Identifier),
    /// Expression in parentheses.
    Expression(Arc<Expression<J>>),
    /// Tuple expression.
    Tuple(Arc<[Expression<J>]>),
    /// Array expression.
    Array(Arc<[Expression<J>]>),
    /// Bounded list of expressions.
    List(Arc<[Expression<J>]>),
    /// Either expression.
    Either(Either<Arc<Expression<J>>, Arc<Expression<J>>>),
    /// Option expression.
    Option(Option<Arc<Expression<J>>>),
    /// Call expression.
    Call(Call<J>),
    /// Match expression.
    Match(Match<J>),
}

/// Call of a user-defined or of a builtin function.
#[derive(Clone, Debug)]
pub struct Call<J: JetHL> {
    name: CallName<J>,
    args: Arc<[Expression<J>]>,
    span: Span,
}

impl<J: JetHL> Call<J> {
    /// Access the name of the call.
    pub fn name(&self) -> &CallName<J> {
        &self.name
    }

    /// Access the arguments of the call.
    pub fn args(&self) -> &Arc<[Expression<J>]> {
        &self.args
    }

    /// Access the span of the call.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl_eq_hash!(Call<J: JetHL>; name, args);

/// Name of a called function.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum CallName<J: JetHL> {
    /// Elements jet.
    Jet(J),
    /// [`Either::unwrap_left`].
    UnwrapLeft(ResolvedType),
    /// [`Either::unwrap_right`].
    UnwrapRight(ResolvedType),
    /// [`Option::is_none`].
    IsNone(ResolvedType),
    /// [`Option::unwrap`].
    Unwrap,
    /// [`assert!`].
    Assert,
    /// [`panic!`] without error message.
    Panic,
    /// [`dbg!`].
    Debug,
    /// Cast from the given source type.
    TypeCast(ResolvedType),
    /// A custom function that was defined previously.
    ///
    /// We effectively copy the function body into every call of the function.
    /// We use [`Arc`] for cheap clones during this process.
    Custom(CustomFunction<J>),
    /// Fold of a bounded list with the given function.
    Fold(CustomFunction<J>, NonZeroPow2Usize),
    /// Fold of an array with the given function.
    ArrayFold(CustomFunction<J>, NonZeroUsize),
    /// Loop over the given function a bounded number of times until it returns success.
    ForWhile(CustomFunction<J>, Pow2Usize),
}

/// Definition of a custom function.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct CustomFunction<J: JetHL> {
    params: Arc<[FunctionParam]>,
    body: Arc<Expression<J>>,
}

impl<J: JetHL> CustomFunction<J> {
    /// Access the identifiers of the parameters of the function.
    pub fn params(&self) -> &[FunctionParam] {
        &self.params
    }

    /// Access the body of the function.
    pub fn body(&self) -> &Expression<J> {
        &self.body
    }

    /// Return a pattern for the parameters of the function.
    pub fn params_pattern(&self) -> Pattern {
        Pattern::tuple(
            self.params()
                .iter()
                .map(FunctionParam::identifier)
                .cloned()
                .map(Pattern::Identifier),
        )
    }
}

/// Parameter of a function.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FunctionParam {
    identifier: Identifier,
    ty: ResolvedType,
}

impl FunctionParam {
    /// Access the identifier of the parameter.
    pub fn identifier(&self) -> &Identifier {
        &self.identifier
    }

    /// Access the type of the parameter.
    pub fn ty(&self) -> &ResolvedType {
        &self.ty
    }
}

/// Match expression.
#[derive(Clone, Debug)]
pub struct Match<J: JetHL> {
    scrutinee: Arc<Expression<J>>,
    left: MatchArm<J>,
    right: MatchArm<J>,
    span: Span,
}

impl<J: JetHL> Match<J> {
    /// Access the expression whose output is destructed in the match statement.
    pub fn scrutinee(&self) -> &Expression<J> {
        &self.scrutinee
    }

    /// Access the branch that handles structural left values.
    pub fn left(&self) -> &MatchArm<J> {
        &self.left
    }

    /// Access the branch that handles structural right values.
    pub fn right(&self) -> &MatchArm<J> {
        &self.right
    }

    /// Access the span of the match statement.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl_eq_hash!(Match<J: JetHL>; scrutinee, left, right);

/// Arm of a [`Match`] expression.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MatchArm<J: JetHL> {
    pattern: MatchPattern,
    expression: Arc<Expression<J>>,
}

impl<J: JetHL> MatchArm<J> {
    /// Access the pattern of the match arm.
    pub fn pattern(&self) -> &MatchPattern {
        &self.pattern
    }

    /// Access the expression of the match arm.
    pub fn expression(&self) -> &Expression<J> {
        &self.expression
    }
}

/// Item when analyzing modules.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ModuleItem<J: JetHL> {
    Ignored,
    Module(Module<J>),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Module<J: JetHL> {
    name: ModuleName,
    assignments: Arc<[ModuleAssignment<J>]>,
    span: Span,
}

impl<J: JetHL> Module<J> {
    /// Access the assignments of the module.
    pub fn assignments(&self) -> &[ModuleAssignment<J>] {
        &self.assignments
    }

    /// Access the span of the module.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModuleAssignment<J: JetHL> {
    name: WitnessName,
    value: Value<J>,
    span: Span,
}

impl<J: JetHL> ModuleAssignment<J> {
    /// Access the assigned witness name.
    pub fn name(&self) -> &WitnessName {
        &self.name
    }

    /// Access the assigned witness value.
    pub fn value(&self) -> &Value<J> {
        &self.value
    }

    /// Access the span of the module.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExprTree<'a, J: JetHL> {
    Expression(&'a Expression<J>),
    Block(&'a [Statement<J>], &'a Option<Arc<Expression<J>>>),
    Statement(&'a Statement<J>),
    Assignment(&'a Assignment<J>),
    Single(&'a SingleExpression<J>),
    Call(&'a Call<J>),
    Match(&'a Match<J>),
}

impl<J: JetHL> TreeLike for ExprTree<'_, J> {
    fn as_node(&self) -> Tree<Self> {
        use SingleExpressionInner as S;

        match self {
            Self::Expression(expr) => match expr.inner() {
                ExpressionInner::Block(statements, maybe_expr) => {
                    Tree::Unary(Self::Block(statements, maybe_expr))
                }
                ExpressionInner::Single(single) => Tree::Unary(Self::Single(single)),
            },
            Self::Block(statements, maybe_expr) => Tree::Nary(
                statements
                    .iter()
                    .map(Self::Statement)
                    .chain(maybe_expr.iter().map(Arc::as_ref).map(Self::Expression))
                    .collect(),
            ),
            Self::Statement(statement) => match statement {
                Statement::Assignment(assignment) => Tree::Unary(Self::Assignment(assignment)),
                Statement::Expression(expression) => Tree::Unary(Self::Expression(expression)),
            },
            Self::Assignment(assignment) => Tree::Unary(Self::Expression(assignment.expression())),
            Self::Single(single) => match single.inner() {
                S::Constant(_)
                | S::Witness(_)
                | S::Parameter(_)
                | S::Variable(_)
                | S::Option(None) => Tree::Nullary,
                S::Expression(l)
                | S::Either(Either::Left(l))
                | S::Either(Either::Right(l))
                | S::Option(Some(l)) => Tree::Unary(Self::Expression(l)),
                S::Tuple(elements) | S::Array(elements) | S::List(elements) => {
                    Tree::Nary(elements.iter().map(Self::Expression).collect())
                }
                S::Call(call) => Tree::Unary(Self::Call(call)),
                S::Match(match_) => Tree::Unary(Self::Match(match_)),
            },
            Self::Call(call) => Tree::Nary(call.args().iter().map(Self::Expression).collect()),
            Self::Match(match_) => Tree::Nary(Arc::new([
                Self::Expression(match_.scrutinee()),
                Self::Expression(match_.left().expression()),
                Self::Expression(match_.right().expression()),
            ])),
        }
    }
}

/// Scope for generating the abstract syntax tree.
///
/// The scope is used for:
/// 1. Assigning types to each variable
/// 2. Resolving type aliases
/// 3. Assigning types to each witness expression
/// 4. Resolving calls to custom functions
#[derive(Clone, Debug, Eq, PartialEq)]
struct Scope<J: JetHL> {
    variables: Vec<HashMap<Identifier, ResolvedType>>,
    aliases: HashMap<AliasName, ResolvedType>,
    parameters: HashMap<WitnessName, ResolvedType>,
    witnesses: HashMap<WitnessName, ResolvedType>,
    functions: HashMap<FunctionName, CustomFunction<J>>,
    is_main: bool,
    call_tracker: CallTracker,
}

impl<J: JetHL> Scope<J> {
    /// Check if the current scope is topmost.
    pub fn is_topmost(&self) -> bool {
        self.variables.is_empty()
    }

    /// Push a new scope onto the stack.
    pub fn push_scope(&mut self) {
        self.variables.push(HashMap::new());
    }

    /// Push the scope of the main function onto the stack.
    ///
    /// ## Panics
    ///
    /// - The current scope is already inside the main function.
    /// - The current scope is not topmost.
    pub fn push_main_scope(&mut self) {
        assert!(!self.is_main, "Already inside main function");
        assert!(self.is_topmost(), "Current scope is not topmost");
        self.push_scope();
        self.is_main = true;
    }

    /// Pop the current scope from the stack.
    ///
    /// ## Panics
    ///
    /// The stack is empty.
    pub fn pop_scope(&mut self) {
        self.variables.pop().expect("Stack is empty");
    }

    /// Pop the scope of the main function from the stack.
    ///
    /// ## Panics
    ///
    /// - The current scope is not inside the main function.
    /// - The current scope is not nested in the topmost scope.
    pub fn pop_main_scope(&mut self) {
        assert!(self.is_main, "Current scope is not inside main function");
        self.pop_scope();
        self.is_main = false;
        assert!(
            self.is_topmost(),
            "Current scope is not nested in topmost scope"
        )
    }

    /// Push a variable onto the current stack.
    ///
    /// ## Panics
    ///
    /// The stack is empty.
    pub fn insert_variable(&mut self, identifier: Identifier, ty: ResolvedType) {
        self.variables
            .last_mut()
            .expect("Stack is empty")
            .insert(identifier, ty);
    }

    /// Get the type of the variable.
    pub fn get_variable(&self, identifier: &Identifier) -> Option<&ResolvedType> {
        self.variables
            .iter()
            .rev()
            .find_map(|scope| scope.get(identifier))
    }

    /// Resolve a type with aliases to a type without aliases.
    ///
    /// ## Errors
    ///
    /// There are any undefined aliases.
    pub fn resolve(&self, ty: &AliasedType) -> Result<ResolvedType, Error> {
        let get_alias =
            |name: &AliasName| -> Option<ResolvedType> { self.aliases.get(name).cloned() };
        ty.resolve(get_alias).map_err(Error::UndefinedAlias)
    }

    /// Push a type alias into the global map.
    ///
    /// ## Errors
    ///
    /// There are any undefined aliases.
    pub fn insert_alias(&mut self, name: AliasName, ty: AliasedType) -> Result<(), Error> {
        let resolved_ty = self.resolve(&ty)?;
        self.aliases.insert(name, resolved_ty);
        Ok(())
    }

    /// Insert a parameter into the global map.
    ///
    /// ## Errors
    ///
    /// A parameter of the same name has already been defined as a different type.
    pub fn insert_parameter(&mut self, name: WitnessName, ty: ResolvedType) -> Result<(), Error> {
        match self.parameters.entry(name.clone()) {
            Entry::Occupied(entry) if entry.get() == &ty => Ok(()),
            Entry::Occupied(entry) => Err(Error::ExpressionTypeMismatch(entry.get().clone(), ty)),
            Entry::Vacant(entry) => {
                entry.insert(ty);
                Ok(())
            }
        }
    }

    /// Insert a witness into the global map.
    ///
    /// ## Errors
    ///
    /// - The current scope is not inside the main function.
    /// - A witness with the same name has already been defined.
    pub fn insert_witness(&mut self, name: WitnessName, ty: ResolvedType) -> Result<(), Error> {
        if !self.is_main {
            return Err(Error::WitnessOutsideMain);
        }

        match self.witnesses.entry(name.clone()) {
            Entry::Occupied(_) => Err(Error::WitnessReused(name)),
            Entry::Vacant(entry) => {
                entry.insert(ty);
                Ok(())
            }
        }
    }

    /// Consume the scope and return its contents:
    ///
    /// 1. The map of parameter types.
    /// 2. The map of witness types.
    /// 3. The function call tracker.
    pub fn destruct(self) -> (Parameters, WitnessTypes, CallTracker) {
        (
            Parameters::from(self.parameters),
            WitnessTypes::from(self.witnesses),
            self.call_tracker,
        )
    }

    /// Insert a custom function into the global map.
    ///
    /// ## Errors
    ///
    /// The function has already been defined.
    pub fn insert_function(
        &mut self,
        name: FunctionName,
        function: CustomFunction<J>,
    ) -> Result<(), Error> {
        match self.functions.entry(name.clone()) {
            Entry::Occupied(_) => Err(Error::FunctionRedefined(name)),
            Entry::Vacant(entry) => {
                entry.insert(function);
                Ok(())
            }
        }
    }

    /// Get the definition of a custom function.
    pub fn get_function(&self, name: &FunctionName) -> Option<&CustomFunction<J>> {
        self.functions.get(name)
    }

    /// Track a call expression with its span.
    pub fn track_call<S: AsRef<Span>>(&mut self, span: &S, name: TrackedCallName) {
        self.call_tracker.track_call(*span.as_ref(), name);
    }
}

impl<J: JetHL> Default for Scope<J> {
    fn default() -> Self {
        Self {
            variables: Vec::new(),
            aliases: HashMap::new(),
            parameters: HashMap::new(),
            witnesses: HashMap::new(),
            functions: HashMap::new(),
            is_main: false,
            call_tracker: CallTracker::default(),
        }
    }
}
/// Part of the abstract syntax tree that can be generated from a precursor in the parse tree.
trait AbstractSyntaxTree<J: JetHL>: Sized {
    /// Component of the parse tree.
    type From;

    /// Analyze a component from the parse tree
    /// and convert it into a component of the abstract syntax tree.
    ///
    /// Check if the analyzed expression is of the expected type.
    /// Statements return no values so their expected type is always unit.
    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError>;
}

impl<J: JetHL> Program<J> {
    pub fn analyze(from: &parse::Program) -> Result<Self, RichError> {
        let unit = ResolvedType::unit();
        let mut scope = Scope::<J>::default();
        let items = from
            .items()
            .iter()
            .map(|s| Item::analyze(s, &unit, &mut scope))
            .collect::<Result<Vec<Item<J>>, RichError>>()?;
        debug_assert!(scope.is_topmost());
        let (parameters, witness_types, call_tracker) = scope.destruct();
        let mut iter = items.into_iter().filter_map(|item| match item {
            Item::Function(Function::Main(expr)) => Some(expr),
            _ => None,
        });
        let main = iter.next().ok_or(Error::MainRequired).with_span(from)?;
        if iter.next().is_some() {
            return Err(Error::FunctionRedefined(FunctionName::main())).with_span(from);
        }
        Ok(Self {
            main,
            parameters,
            witness_types,
            call_tracker: Arc::new(call_tracker),
        })
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Item<J> {
    type From = parse::Item;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Items cannot return anything");
        assert!(scope.is_topmost(), "Items live in the topmost scope only");

        match from {
            parse::Item::TypeAlias(alias) => {
                scope
                    .insert_alias(alias.name().clone(), alias.ty().clone())
                    .with_span(alias)?;
                Ok(Self::TypeAlias)
            }
            parse::Item::Function(function) => {
                Function::analyze(function, ty, scope).map(Self::Function)
            }
            parse::Item::Module => Ok(Self::Module),
        }
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Function<J> {
    type From = parse::Function;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Function definitions cannot return anything");
        assert!(scope.is_topmost(), "Items live in the topmost scope only");

        if from.name().as_inner() != "main" {
            let params = from
                .params()
                .iter()
                .map(|param| {
                    let identifier = param.identifier().clone();
                    let ty = scope.resolve(param.ty())?;
                    Ok(FunctionParam { identifier, ty })
                })
                .collect::<Result<Arc<[FunctionParam]>, Error>>()
                .with_span(from)?;
            let ret = from
                .ret()
                .as_ref()
                .map(|aliased| scope.resolve(aliased).with_span(from))
                .transpose()?
                .unwrap_or_else(ResolvedType::unit);
            scope.push_scope();
            for param in params.iter() {
                scope.insert_variable(param.identifier().clone(), param.ty().clone());
            }
            let body = Expression::analyze(from.body(), &ret, scope).map(Arc::new)?;
            scope.pop_scope();
            debug_assert!(scope.is_topmost());
            let function = CustomFunction { params, body };
            scope
                .insert_function(from.name().clone(), function)
                .with_span(from)?;

            return Ok(Self::Custom);
        }

        if !from.params().is_empty() {
            return Err(Error::MainNoInputs).with_span(from);
        }
        if let Some(aliased) = from.ret() {
            let resolved = scope.resolve(aliased).with_span(from)?;
            if !resolved.is_unit() {
                return Err(Error::MainNoOutput).with_span(from);
            }
        }

        scope.push_main_scope();
        let body = Expression::analyze(from.body(), ty, scope)?;
        scope.pop_main_scope();
        Ok(Self::Main(body))
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Statement<J> {
    type From = parse::Statement;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Statements cannot return anything");
        match from {
            parse::Statement::Assignment(assignment) => {
                Assignment::analyze(assignment, ty, scope).map(Self::Assignment)
            }
            parse::Statement::Expression(expression) => {
                Expression::analyze(expression, ty, scope).map(Self::Expression)
            }
        }
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Assignment<J> {
    type From = parse::Assignment;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Assignments cannot return anything");
        // The assignment is a statement that returns nothing.
        //
        // However, the expression evaluated in the assignment does have a type,
        // namely the type specified in the assignment.
        let ty_expr = scope.resolve(from.ty()).with_span(from)?;
        let expression = Expression::analyze(from.expression(), &ty_expr, scope)?;
        let typed_variables = from.pattern().is_of_type(&ty_expr).with_span(from)?;
        for (identifier, ty) in typed_variables {
            scope.insert_variable(identifier, ty);
        }

        Ok(Self {
            pattern: from.pattern().clone(),
            expression,
            span: *from.as_ref(),
        })
    }
}

impl<J: JetHL> Expression<J> {
    /// Analyze an expression from the parse tree in a const context without predefined variables.
    ///
    /// Check if the expression is of the given type.
    ///
    /// ## Const evaluation
    ///
    /// The returned expression might not be evaluable at compile time.
    /// The details depend on the current state of the SimplicityHL compiler.
    pub fn analyze_const(from: &parse::Expression, ty: &ResolvedType) -> Result<Self, RichError> {
        let mut empty_scope = Scope::<J>::default();
        Self::analyze(from, ty, &mut empty_scope)
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Expression<J> {
    type From = parse::Expression;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        match from.inner() {
            parse::ExpressionInner::Single(single) => {
                let ast_single = SingleExpression::analyze(single, ty, scope)?;
                Ok(Self {
                    ty: ty.clone(),
                    inner: ExpressionInner::Single(ast_single),
                    span: *from.as_ref(),
                })
            }
            parse::ExpressionInner::Block(statements, expression) => {
                scope.push_scope();
                let ast_statements = statements
                    .iter()
                    .map(|s| Statement::analyze(s, &ResolvedType::unit(), scope))
                    .collect::<Result<Arc<[Statement<J>]>, RichError>>()?;
                let ast_expression = match expression {
                    Some(expression) => Expression::analyze(expression, ty, scope)
                        .map(Arc::new)
                        .map(Some),
                    None if ty.is_unit() => Ok(None),
                    None => Err(Error::ExpressionTypeMismatch(
                        ty.clone(),
                        ResolvedType::unit(),
                    ))
                    .with_span(from),
                }?;
                scope.pop_scope();

                Ok(Self {
                    ty: ty.clone(),
                    inner: ExpressionInner::Block(ast_statements, ast_expression),
                    span: *from.as_ref(),
                })
            }
        }
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for SingleExpression<J> {
    type From = parse::SingleExpression;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        let inner = match from.inner() {
            parse::SingleExpressionInner::Boolean(bit) => {
                if !ty.is_boolean() {
                    return Err(Error::ExpressionTypeMismatch(
                        ty.clone(),
                        ResolvedType::boolean(),
                    ))
                    .with_span(from);
                }
                SingleExpressionInner::Constant(Value::from(*bit))
            }
            parse::SingleExpressionInner::Decimal(decimal) => {
                let ty = ty
                    .as_integer()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                UIntValue::parse_decimal(decimal, ty)
                    .with_span(from)
                    .map(Value::from)
                    .map(SingleExpressionInner::Constant)?
            }
            parse::SingleExpressionInner::Binary(bits) => {
                let ty = ty
                    .as_integer()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                let value = UIntValue::parse_binary(bits, ty).with_span(from)?;
                SingleExpressionInner::Constant(Value::from(value))
            }
            parse::SingleExpressionInner::Hexadecimal(bytes) => {
                let value = Value::parse_hexadecimal(bytes, ty).with_span(from)?;
                SingleExpressionInner::Constant(value)
            }
            parse::SingleExpressionInner::Witness(name) => {
                scope
                    .insert_witness(name.clone(), ty.clone())
                    .with_span(from)?;
                SingleExpressionInner::Witness(name.clone())
            }
            parse::SingleExpressionInner::Parameter(name) => {
                scope
                    .insert_parameter(name.shallow_clone(), ty.clone())
                    .with_span(from)?;
                SingleExpressionInner::Parameter(name.shallow_clone())
            }
            parse::SingleExpressionInner::Variable(identifier) => {
                let bound_ty = scope
                    .get_variable(identifier)
                    .ok_or(Error::UndefinedVariable(identifier.clone()))
                    .with_span(from)?;
                if ty != bound_ty {
                    return Err(Error::ExpressionTypeMismatch(ty.clone(), bound_ty.clone()))
                        .with_span(from);
                }
                scope.insert_variable(identifier.clone(), ty.clone());
                SingleExpressionInner::Variable(identifier.clone())
            }
            parse::SingleExpressionInner::Expression(parse) => {
                Expression::analyze(parse, ty, scope)
                    .map(Arc::new)
                    .map(SingleExpressionInner::Expression)?
            }
            parse::SingleExpressionInner::Tuple(tuple) => {
                let types = ty
                    .as_tuple()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                if tuple.len() != types.len() {
                    return Err(Error::ExpressionUnexpectedType(ty.clone())).with_span(from);
                }
                tuple
                    .iter()
                    .zip(types.iter())
                    .map(|(el_parse, el_ty)| Expression::analyze(el_parse, el_ty, scope))
                    .collect::<Result<Arc<[Expression<J>]>, RichError>>()
                    .map(SingleExpressionInner::Tuple)?
            }
            parse::SingleExpressionInner::Array(array) => {
                let (el_ty, size) = ty
                    .as_array()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                if array.len() != size {
                    return Err(Error::ExpressionUnexpectedType(ty.clone())).with_span(from);
                }
                array
                    .iter()
                    .map(|el_parse| Expression::analyze(el_parse, el_ty, scope))
                    .collect::<Result<Arc<[Expression<J>]>, RichError>>()
                    .map(SingleExpressionInner::Array)?
            }
            parse::SingleExpressionInner::List(list) => {
                let (el_ty, bound) = ty
                    .as_list()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                if bound.get() <= list.len() {
                    return Err(Error::ExpressionUnexpectedType(ty.clone())).with_span(from);
                }
                list.iter()
                    .map(|e| Expression::analyze(e, el_ty, scope))
                    .collect::<Result<Arc<[Expression<J>]>, RichError>>()
                    .map(SingleExpressionInner::List)?
            }
            parse::SingleExpressionInner::Either(either) => {
                let (ty_l, ty_r) = ty
                    .as_either()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                match either {
                    Either::Left(parse_l) => Expression::analyze(parse_l, ty_l, scope)
                        .map(Arc::new)
                        .map(Either::Left),
                    Either::Right(parse_r) => Expression::analyze(parse_r, ty_r, scope)
                        .map(Arc::new)
                        .map(Either::Right),
                }
                .map(SingleExpressionInner::Either)?
            }
            parse::SingleExpressionInner::Option(maybe_parse) => {
                let ty = ty
                    .as_option()
                    .ok_or(Error::ExpressionUnexpectedType(ty.clone()))
                    .with_span(from)?;
                match maybe_parse {
                    Some(parse) => {
                        Some(Expression::analyze(parse, ty, scope).map(Arc::new)).transpose()
                    }
                    None => Ok(None),
                }
                .map(SingleExpressionInner::Option)?
            }
            parse::SingleExpressionInner::Call(call) => {
                Call::analyze(call, ty, scope).map(SingleExpressionInner::Call)?
            }
            parse::SingleExpressionInner::Match(match_) => {
                Match::analyze(match_, ty, scope).map(SingleExpressionInner::Match)?
            }
        };

        Ok(Self {
            inner,
            ty: ty.clone(),
            span: *from.as_ref(),
        })
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Call<J> {
    type From = parse::Call;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        fn check_argument_types(
            parse_args: &[parse::Expression],
            expected_tys: &[ResolvedType],
        ) -> Result<(), Error> {
            if parse_args.len() == expected_tys.len() {
                Ok(())
            } else {
                Err(Error::InvalidNumberOfArguments(
                    expected_tys.len(),
                    parse_args.len(),
                ))
            }
        }

        fn check_output_type(
            observed_ty: &ResolvedType,
            expected_ty: &ResolvedType,
        ) -> Result<(), Error> {
            if observed_ty == expected_ty {
                Ok(())
            } else {
                Err(Error::ExpressionTypeMismatch(
                    expected_ty.clone(),
                    observed_ty.clone(),
                ))
            }
        }

        fn analyze_arguments<J: JetHL>(
            parse_args: &[parse::Expression],
            args_tys: &[ResolvedType],
            scope: &mut Scope<J>,
        ) -> Result<Arc<[Expression<J>]>, RichError> {
            let args = parse_args
                .iter()
                .zip(args_tys.iter())
                .map(|(arg_parse, arg_ty)| Expression::analyze(arg_parse, arg_ty, scope))
                .collect::<Result<Arc<[Expression<J>]>, RichError>>()?;
            Ok(args)
        }

        let name = CallName::analyze(from, ty, scope)?;
        let args = match name.clone() {
            CallName::Jet(jet) => {
                let args_tys = J::source_type(jet)
                    .iter()
                    .map(AliasedType::resolve_builtin)
                    .collect::<Result<Vec<ResolvedType>, AliasName>>()
                    .map_err(Error::UndefinedAlias)
                    .with_span(from)?;
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let out_ty = J::target_type(jet)
                    .resolve_builtin()
                    .map_err(Error::UndefinedAlias)
                    .with_span(from)?;
                check_output_type(&out_ty, ty).with_span(from)?;
                scope.track_call(from, TrackedCallName::Jet);
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::UnwrapLeft(right_ty) => {
                let args_tys = [ResolvedType::either(ty.clone(), right_ty)];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let args = analyze_arguments(from.args(), &args_tys, scope)?;
                let [arg_ty] = args_tys;
                scope.track_call(from, TrackedCallName::UnwrapLeft(arg_ty));
                args
            }
            CallName::UnwrapRight(left_ty) => {
                let args_tys = [ResolvedType::either(left_ty, ty.clone())];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let args = analyze_arguments(from.args(), &args_tys, scope)?;
                let [arg_ty] = args_tys;
                scope.track_call(from, TrackedCallName::UnwrapRight(arg_ty));
                args
            }
            CallName::IsNone(some_ty) => {
                let args_tys = [ResolvedType::option(some_ty)];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let out_ty = ResolvedType::boolean();
                check_output_type(&out_ty, ty).with_span(from)?;
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::Unwrap => {
                let args_tys = [ResolvedType::option(ty.clone())];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                scope.track_call(from, TrackedCallName::Unwrap);
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::Assert => {
                let args_tys = [ResolvedType::boolean()];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let out_ty = ResolvedType::unit();
                check_output_type(&out_ty, ty).with_span(from)?;
                scope.track_call(from, TrackedCallName::Assert);
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::Panic => {
                let args_tys = [];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                // panic! allows every output type because it will never return anything
                scope.track_call(from, TrackedCallName::Panic);
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::Debug => {
                let args_tys = [ty.clone()];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                let args = analyze_arguments(from.args(), &args_tys, scope)?;
                let [arg_ty] = args_tys;
                scope.track_call(from, TrackedCallName::Debug(arg_ty));
                args
            }
            CallName::TypeCast(source) => {
                if StructuralType::from(&source) != StructuralType::from(ty) {
                    return Err(Error::InvalidCast(source, ty.clone())).with_span(from);
                }

                let args_tys = [source];
                check_argument_types(from.args(), &args_tys).with_span(from)?;
                analyze_arguments(from.args(), &args_tys, scope)?
            }
            CallName::Custom(function) => {
                let args_ty = function
                    .params()
                    .iter()
                    .map(FunctionParam::ty)
                    .cloned()
                    .collect::<Vec<ResolvedType>>();
                check_argument_types(from.args(), &args_ty).with_span(from)?;
                let out_ty = function.body().ty();
                check_output_type(out_ty, ty).with_span(from)?;
                analyze_arguments(from.args(), &args_ty, scope)?
            }
            CallName::Fold(function, bound) => {
                // A list fold has the signature:
                //   fold::<f, N>(list: List<E, N>, initial_accumulator: A) -> A
                // where
                //   fn f(element: E, accumulator: A) -> A
                let element_ty = function.params().first().expect("foldable function").ty();
                let list_ty = ResolvedType::list(element_ty.clone(), bound);
                let accumulator_ty = function
                    .params()
                    .get(1)
                    .expect("foldable function")
                    .ty()
                    .clone();
                let args_ty = [list_ty, accumulator_ty];

                check_argument_types(from.args(), &args_ty).with_span(from)?;
                let out_ty = function.body().ty();
                check_output_type(out_ty, ty).with_span(from)?;
                analyze_arguments(from.args(), &args_ty, scope)?
            }
            CallName::ArrayFold(function, size) => {
                // An array fold has the signature:
                //   array_fold::<f, N>(array: [E; N], initial_accumulator: A) -> A
                // where
                //   fn f(element: E, accumulator: A) -> A
                let element_ty = function.params().first().expect("foldable function").ty();
                let array_ty = ResolvedType::array(element_ty.clone(), size.get());
                let accumulator_ty = function
                    .params()
                    .get(1)
                    .expect("foldable function")
                    .ty()
                    .clone();
                let args_ty = [array_ty, accumulator_ty];

                check_argument_types(from.args(), &args_ty).with_span(from)?;
                let out_ty = function.body().ty();
                check_output_type(out_ty, ty).with_span(from)?;
                analyze_arguments(from.args(), &args_ty, scope)?
            }
            CallName::ForWhile(function, _bit_width) => {
                // A for-while loop has the signature:
                //   for_while::<f>(initial_accumulator: A, readonly_context: C) -> Either<B, A>
                // where
                //   fn f(accumulator: A, readonly_context: C, counter: u{N}) -> Either<B, A>
                //   N is a power of two
                let accumulator_ty = function
                    .params()
                    .first()
                    .expect("loopable function")
                    .ty()
                    .clone();
                let context_ty = function
                    .params()
                    .get(1)
                    .expect("loopable function")
                    .ty()
                    .clone();
                let args_ty = [accumulator_ty, context_ty];

                check_argument_types(from.args(), &args_ty).with_span(from)?;
                let out_ty = function.body().ty();
                check_output_type(out_ty, ty).with_span(from)?;
                analyze_arguments(from.args(), &args_ty, scope)?
            }
        };

        Ok(Self {
            name,
            args,
            span: *from.as_ref(),
        })
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for CallName<J> {
    // Take parse::Call, so we have access to the span for pretty errors
    type From = parse::Call;

    fn analyze(
        from: &Self::From,
        _ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        match from.name() {
            parse::CallName::Jet(name) => match J::from_str(name.as_inner()) {
                Ok(jet) => Ok(Self::Jet(J::filter(jet, name).with_span(from)?)),
                Err(_) => Err(Error::JetDoesNotExist(name.clone())).with_span(from),
            },
            parse::CallName::UnwrapLeft(right_ty) => scope
                .resolve(right_ty)
                .map(Self::UnwrapLeft)
                .with_span(from),
            parse::CallName::UnwrapRight(left_ty) => scope
                .resolve(left_ty)
                .map(Self::UnwrapRight)
                .with_span(from),
            parse::CallName::IsNone(some_ty) => {
                scope.resolve(some_ty).map(Self::IsNone).with_span(from)
            }
            parse::CallName::Unwrap => Ok(Self::Unwrap),
            parse::CallName::Assert => Ok(Self::Assert),
            parse::CallName::Panic => Ok(Self::Panic),
            parse::CallName::Debug => Ok(Self::Debug),
            parse::CallName::TypeCast(target) => {
                scope.resolve(target).map(Self::TypeCast).with_span(from)
            }
            parse::CallName::Custom(name) => scope
                .get_function(name)
                .cloned()
                .map(Self::Custom)
                .ok_or(Error::FunctionUndefined(name.clone()))
                .with_span(from),
            parse::CallName::ArrayFold(name, size) => {
                let function = scope
                    .get_function(name)
                    .cloned()
                    .ok_or(Error::FunctionUndefined(name.clone()))
                    .with_span(from)?;
                // A function that is used in a array fold has the signature:
                //   fn f(element: E, accumulator: A) -> A
                if function.params().len() != 2 || function.params()[1].ty() != function.body().ty()
                {
                    Err(Error::FunctionNotFoldable(name.clone())).with_span(from)
                } else {
                    Ok(Self::ArrayFold(function, *size))
                }
            }
            parse::CallName::Fold(name, bound) => {
                let function = scope
                    .get_function(name)
                    .cloned()
                    .ok_or(Error::FunctionUndefined(name.clone()))
                    .with_span(from)?;
                // A function that is used in a list fold has the signature:
                //   fn f(element: E, accumulator: A) -> A
                if function.params().len() != 2 || function.params()[1].ty() != function.body().ty()
                {
                    Err(Error::FunctionNotFoldable(name.clone())).with_span(from)
                } else {
                    Ok(Self::Fold(function, *bound))
                }
            }
            parse::CallName::ForWhile(name) => {
                let function = scope
                    .get_function(name)
                    .cloned()
                    .ok_or(Error::FunctionUndefined(name.clone()))
                    .with_span(from)?;
                // A function that is used in a for-while loop has the signature:
                //   fn f(accumulator: A, readonly_context: C, counter: u{N}) -> Either<B, A>
                // where
                //   N is a power of two
                if function.params().len() != 3 {
                    return Err(Error::FunctionNotLoopable(name.clone())).with_span(from);
                }
                match function.body().ty().as_either() {
                    Some((_, out_r)) if out_r == function.params().first().unwrap().ty() => {}
                    _ => {
                        return Err(Error::FunctionNotLoopable(name.clone())).with_span(from);
                    }
                }
                // Disable loops for u32 or higher since no one will want to run
                // 2^32 = 4294967296 ≈ 4 billion iterations.
                // The resulting Simplicity program will not fit into a Bitcoin block.
                match function.params().get(2).unwrap().ty().as_integer() {
                    Some(
                        int_ty @ (UIntType::U1
                        | UIntType::U2
                        | UIntType::U4
                        | UIntType::U8
                        | UIntType::U16),
                    ) => Ok(Self::ForWhile(function, int_ty.bit_width())),
                    _ => Err(Error::FunctionNotLoopable(name.clone())).with_span(from),
                }
            }
        }
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Match<J> {
    type From = parse::Match;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        let scrutinee_ty = from.scrutinee_type();
        let scrutinee_ty = scope.resolve(&scrutinee_ty).with_span(from)?;
        let scrutinee =
            Expression::analyze(from.scrutinee(), &scrutinee_ty, scope).map(Arc::new)?;

        scope.push_scope();
        if let Some((id_l, ty_l)) = from.left().pattern().as_typed_variable() {
            let ty_l = scope.resolve(ty_l).with_span(from)?;
            scope.insert_variable(id_l.clone(), ty_l);
        }
        let ast_l = Expression::analyze(from.left().expression(), ty, scope).map(Arc::new)?;
        scope.pop_scope();
        scope.push_scope();
        if let Some((id_r, ty_r)) = from.right().pattern().as_typed_variable() {
            let ty_r = scope.resolve(ty_r).with_span(from)?;
            scope.insert_variable(id_r.clone(), ty_r);
        }
        let ast_r = Expression::analyze(from.right().expression(), ty, scope).map(Arc::new)?;
        scope.pop_scope();

        Ok(Self {
            scrutinee,
            left: MatchArm {
                pattern: from.left().pattern().clone(),
                expression: ast_l,
            },
            right: MatchArm {
                pattern: from.right().pattern().clone(),
                expression: ast_r,
            },
            span: *from.as_ref(),
        })
    }
}

fn analyze_named_module<J: JetHL>(
    name: ModuleName,
    from: &parse::ModuleProgram,
) -> Result<HashMap<WitnessName, Value<J>>, RichError> {
    let unit = ResolvedType::unit();
    let mut scope = Scope::<J>::default();
    let items = from
        .items()
        .iter()
        .map(|s| ModuleItem::analyze(s, &unit, &mut scope))
        .collect::<Result<Vec<ModuleItem<J>>, RichError>>()?;
    debug_assert!(scope.is_topmost());
    let mut iter = items.into_iter().filter_map(|item| match item {
        ModuleItem::Module(module) if module.name == name => Some(module),
        _ => None,
    });
    let Some(witness_module) = iter.next() else {
        return Ok(HashMap::new()); // "not present" is equivalent to empty
    };
    if iter.next().is_some() {
        return Err(Error::ModuleRedefined(name)).with_span(from);
    }
    let mut map = HashMap::new();
    for assignment in witness_module.assignments() {
        if map.contains_key(assignment.name()) {
            return Err(Error::WitnessReassigned(assignment.name().shallow_clone()))
                .with_span(assignment);
        }
        map.insert(
            assignment.name().shallow_clone(),
            assignment.value().clone(),
        );
    }
    Ok(map)
}

impl<J: JetHL> WitnessValues<J> {
    pub fn analyze(from: &parse::ModuleProgram) -> Result<Self, RichError> {
        analyze_named_module::<J>(ModuleName::witness(), from).map(Self::from)
    }
}

impl<J: JetHL> crate::witness::Arguments<J> {
    pub fn analyze(from: &parse::ModuleProgram) -> Result<Self, RichError> {
        analyze_named_module::<J>(ModuleName::param(), from).map(Self::from)
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for ModuleItem<J> {
    type From = parse::ModuleItem;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Items cannot return anything");
        assert!(scope.is_topmost(), "Items live in the topmost scope only");
        match from {
            parse::ModuleItem::Ignored => Ok(Self::Ignored),
            parse::ModuleItem::Module(witness_module) => {
                Module::analyze(witness_module, ty, scope).map(Self::Module)
            }
        }
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for Module<J> {
    type From = parse::Module;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Modules cannot return anything");
        assert!(scope.is_topmost(), "Modules live in the topmost scope only");
        let assignments = from
            .assignments()
            .iter()
            .map(|s| ModuleAssignment::analyze(s, ty, scope))
            .collect::<Result<Arc<[ModuleAssignment<J>]>, RichError>>()?;
        debug_assert!(scope.is_topmost());

        Ok(Self {
            name: from.name().shallow_clone(),
            span: *from.as_ref(),
            assignments,
        })
    }
}

impl<J: JetHL> AbstractSyntaxTree<J> for ModuleAssignment<J> {
    type From = parse::ModuleAssignment;

    fn analyze(
        from: &Self::From,
        ty: &ResolvedType,
        scope: &mut Scope<J>,
    ) -> Result<Self, RichError> {
        assert!(ty.is_unit(), "Assignments cannot return anything");
        let ty_expr = scope.resolve(from.ty()).with_span(from)?;
        let expression = Expression::analyze(from.expression(), &ty_expr, scope)?;
        let value = Value::from_const_expr(&expression)
            .ok_or(Error::ExpressionUnexpectedType(ty_expr.clone()))
            .with_span(from.expression())?;

        Ok(Self {
            name: from.name().clone(),
            value,
            span: *from.as_ref(),
        })
    }
}

impl<J: JetHL> AsRef<Span> for Assignment<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for Expression<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for SingleExpression<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for Call<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for Match<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for Module<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}

impl<J: JetHL> AsRef<Span> for ModuleAssignment<J> {
    fn as_ref(&self) -> &Span {
        &self.span
    }
}
