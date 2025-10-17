use crate::code::Instructions;
use crate::object::Object;

#[derive(Clone, Debug)]
pub struct Frame {
    pub func: Object,
    pub ip: usize,
    pub base_pointer: usize,
}

impl Frame {
    pub fn new(func: Object, base_pointer: usize) -> Self {
        Frame {
            func,
            ip: 0,
            base_pointer,
        }
    }

    pub fn instructions(&self) -> &Instructions {
        match &self.func {
            Object::CompiledFunction(instructions, _, _) => instructions,
            Object::Closure(closure) => &closure.func.instructions,
            _ => panic!("Expected CompiledFunction or Closure object"),
        }
    }
}
