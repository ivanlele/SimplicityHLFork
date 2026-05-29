use std::sync::OnceLock;
use std::{io::Write, path::Path};

use dlopen2::wrapper::{Container, WrapperApi};

use simplicity::{
    jet::{type_name::TypeName, Jet},
    BitIter, BitWriter, Cmr, Cost,
};

use crate::ast::JetHinter;
use crate::jet::JetHL;
use crate::types::AliasedType;

static EXTERNAL_JET_LIB: OnceLock<Container<ExternalJetLib>> = OnceLock::new();

pub fn init_external_jet_lib(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let container = unsafe { Container::load(Path::new(path))? };

    if EXTERNAL_JET_LIB.set(container).is_err() {
        return Err("Failed to set external jet lib, it may have already been initialized".into());
    }

    Ok(())
}

fn external_jet_lib() -> &'static Container<ExternalJetLib> {
    EXTERNAL_JET_LIB
        .get()
        .expect("External jet lib is not initialized. Please call init_external_jet_lib first.")
}

#[derive(WrapperApi)]
pub struct ExternalJetLib {
    cmr: fn(jet: ExternalJet) -> Cmr,
    source_ty: fn(jet: ExternalJet) -> TypeName,
    target_ty: fn(jet: ExternalJet) -> TypeName,
    encode: fn(jet: ExternalJet, w: &mut BitWriter<&mut dyn Write>) -> std::io::Result<usize>,
    cost: fn(jet: ExternalJet) -> Cost,
    parse: fn(s: &str) -> Result<ExternalJet, simplicity::Error>,
    display: fn(jet: ExternalJet) -> String,

    source_type: fn(jet: ExternalJet) -> Vec<AliasedType>,
    target_type: fn(jet: ExternalJet) -> AliasedType,
    is_disabled: fn(jet: ExternalJet) -> bool,

    verify: fn() -> ExternalJet,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ExternalJet {
    pub index: u64,
}

impl ExternalJet {
    pub fn new(index: u64) -> Self {
        Self { index }
    }
}

impl Jet for ExternalJet {
    fn cmr(&self) -> Cmr {
        let container = external_jet_lib();
        (container.cmr)(*self)
    }

    fn source_ty(&self) -> TypeName {
        let container = external_jet_lib();
        (container.source_ty)(*self)
    }

    fn target_ty(&self) -> TypeName {
        let container = external_jet_lib();
        (container.target_ty)(*self)
    }

    fn encode(&self, w: &mut BitWriter<&mut dyn Write>) -> std::io::Result<usize> {
        let container = external_jet_lib();
        (container.encode)(*self, w)
    }

    fn decode<I: Iterator<Item = u8>>(
        _bits: &mut BitIter<I>,
    ) -> Result<Self, simplicity::decode::Error>
    where
        Self: Sized,
    {
        unimplemented!("Decoding is not implemented for ExternalJet for now")
    }

    fn cost(&self) -> Cost {
        let container = external_jet_lib();
        (container.cost)(*self)
    }

    fn parse(s: &str) -> Result<Self, simplicity::Error>
    where
        Self: Sized,
    {
        let container = external_jet_lib();
        (container.parse)(s)
    }
}

impl std::fmt::Display for ExternalJet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let container = external_jet_lib();
        let display_str = (container.display)(*self);
        write!(f, "{}", display_str)
    }
}

impl JetHL for ExternalJet {
    fn source_type(&self) -> Vec<AliasedType> {
        let container = external_jet_lib();
        (container.source_type)(*self)
    }

    fn target_type(&self) -> AliasedType {
        let container = external_jet_lib();
        (container.target_type)(*self)
    }

    fn is_disabled(&self) -> bool {
        let container = external_jet_lib();
        (container.is_disabled)(*self)
    }

    fn clone_box(&self) -> Box<dyn JetHL> {
        Box::new(*self)
    }

    fn as_jet(&self) -> &dyn Jet {
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExternalJetHinter;

impl ExternalJetHinter {
    pub fn new() -> Self {
        Self
    }
}

impl JetHinter for ExternalJetHinter {
    fn parse_jet(&self, name: &str) -> Option<Box<dyn JetHL>> {
        let container = external_jet_lib();
        match (container.parse)(name) {
            Ok(jet) => Some(Box::new(jet)),
            Err(_) => None,
        }
    }

    fn construct_verify(&self) -> Box<dyn JetHL> {
        let container = external_jet_lib();
        let jet = (container.verify)();
        Box::new(jet)
    }

    fn clone_box(&self) -> Box<dyn JetHinter> {
        Box::new(ExternalJetHinter)
    }
}
