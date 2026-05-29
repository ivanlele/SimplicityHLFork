use std::io::Write;

use simplicityhl::{
    jet::JetHL,
    simplicity::{
        jet::{type_name::TypeName, Jet},
        BitWriter, Cmr, Cost,
    },
    types::AliasedType,
};

use crate::jet::{ExternalJet, HappyJet};

mod jet;

#[no_mangle]
pub fn cmr(jet: ExternalJet) -> Cmr {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.cmr()
}

#[no_mangle]
pub fn source_ty(jet: ExternalJet) -> TypeName {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.source_ty()
}

#[no_mangle]
pub fn target_ty(jet: ExternalJet) -> TypeName {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.target_ty()
}

#[no_mangle]
pub fn encode(jet: ExternalJet, w: &mut dyn Write) -> std::io::Result<usize> {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    let mut bit_writer = BitWriter::new(w);

    jet.encode(&mut bit_writer)
}

#[no_mangle]
pub fn cost(jet: ExternalJet) -> Cost {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.cost()
}

#[no_mangle]
pub fn parse(s: &str) -> Result<ExternalJet, simplicityhl::simplicity::Error> {
    HappyJet::parse(s).map(|jet| ExternalJet { index: jet.index() })
}
#[no_mangle]
pub fn display(jet: ExternalJet) -> String {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.to_string()
}

#[no_mangle]
pub fn source_type(jet: ExternalJet) -> Vec<AliasedType> {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.source_type()
}

#[no_mangle]
pub fn target_type(jet: ExternalJet) -> AliasedType {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.target_type()
}

#[no_mangle]
pub fn is_disabled(jet: ExternalJet) -> bool {
    let jet = HappyJet::from_index(jet.index).expect("invalid jet index");

    jet.is_disabled()
}

#[no_mangle]
pub fn verify() -> ExternalJet {
    let jet = HappyJet::Verify;

    ExternalJet { index: jet.index() }
}
