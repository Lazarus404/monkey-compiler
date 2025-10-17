pub mod frame;
use std::rc::Rc;
use std::cell::RefCell;
use crate::object::Object;
use crate::object::builtins::BUILTINS;
use crate::vm::frame::Frame;
use crate::compiler::Bytecode;
use crate::code::{
    OPCONSTANT,
    OPADD,
    OPSUB,
    OPMULT,
    OPDIV,
    OPTRUE,
    OPFALSE,
    OPGREATERTHAN,
    OPGREATERTHANEQUAL,
    OPLESSTHAN,
    OPLESSTHANEQUAL,
    OPEQUAL,
    OPNOTEQUAL,
    OPMINUS,
    OPPLUS,
    OPBANG,
    OPPOP,
    OPJUMPELSE,
    OPJUMP,
    OPNULL,
    OPGETGLOBAL,
    OPSETGLOBAL,
    OPARRAY,
    OPHASH,
    OPINDEX,
    OPCALL,
    OPRETURNVALUE,
    OPRETURN,
    OPBUILTIN,
    OPGETLOCAL,
    OPSETLOCAL,
    OPCLOSURE,
    OPGETFREE,
    OPCURRENTCLOSURE,
    read_u16, 
};

pub const STACK_SIZE: usize = 2048;
pub const GLOBAL_SIZE: usize = 65536;
pub const MAX_FRAMES: usize = 1024;

pub const TRUE: Object = Object::Bool(true);
pub const FALSE: Object = Object::Bool(false);


pub struct VM {
    pub constants: Vec<Object>,
    pub globals: Rc<RefCell<Vec<Object>>>,

    stack: Vec<Object>,
    sp: usize, // Always points to the next value. Top of stack is stack[sp-1]

    frames: Vec<Frame>,
    frames_index: usize,
}

impl VM {
    pub fn new(bytecode: &Bytecode) -> Self {
        let main_fn = Object::CompiledFunction(bytecode.instructions.clone(), 0, 0);
        let main_frame = Frame::new(main_fn, 0);

        let mut frames = Vec::with_capacity(MAX_FRAMES);
        frames.push(main_frame);

        VM {
            constants: bytecode.constants.clone(),
            globals: Rc::new(RefCell::new(vec![Object::Null; GLOBAL_SIZE])),
            stack: vec![Object::Null; STACK_SIZE],
            sp: 0,
            frames,
            frames_index: 1,
        }
    }

    pub fn new_with_globals_store(
        bytecode: Bytecode,
        globals: Rc<RefCell<Vec<Object>>>,
    ) -> Self {
        let main_fn = Object::CompiledFunction(bytecode.instructions.clone(), 0, 0);
        let main_frame = Frame::new(main_fn, 0);

        let mut frames = Vec::with_capacity(MAX_FRAMES);
        frames.push(main_frame.clone());

        VM {
            constants: bytecode.constants,
            globals: globals,
            stack: vec![Object::Null; STACK_SIZE],
            sp: 0,
            frames: frames,
            frames_index: 1,
        }

    }

    pub fn stack_top(&self) -> Option<&Object> {
        if self.sp == 0 {
            None
        } else {
            self.stack.get(self.sp - 1)
        }
    }

    pub fn run(&mut self) -> Result<(), String> {
        while self.current_frame().ip < self.current_frame().instructions().len() {
            let ip = self.current_frame().ip;
            let ins = self.current_frame().instructions();
            let op = ins[ip];

            match op {
                OPCONSTANT => {
                    let const_index = read_u16(&self.current_frame().instructions().0[ip + 1..]) as usize;
                    self.current_frame_mut().ip += 2;
                    let obj = self.constants[const_index].clone();
                    if let Err(err) = self.push(obj) {
                        return Err(err);
                    }
                }
                OPADD | OPSUB | OPMULT | OPDIV => {
                    if let Err(err) = self.execute_binary_operation(op) {
                        return Err(err);
                    }
                }
                OPTRUE => {
                    self.push(TRUE)?;
                }
                OPFALSE => {
                    self.push(FALSE)?;
                }
                OPEQUAL | OPNOTEQUAL | OPGREATERTHAN | OPLESSTHAN | OPGREATERTHANEQUAL | OPLESSTHANEQUAL => {
                    if let Err(err) = self.execute_comparison(op) {
                        return Err(err);
                    }
                }
                OPMINUS => {
                    if let Err(err) = self.execute_minus_operator() {
                        return Err(err);
                    }
                }
                OPPLUS => {
                    if let Err(err) = self.execute_plus_operator() {
                        return Err(err);
                    }
                }
                OPBANG => {
                    if let Err(err) = self.execute_bang_operator() {
                        return Err(err);
                    }
                }
                OPPOP => {
                    self.pop();
                }
                OPJUMPELSE => {
                    let pos = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip += 2;
                    let condition = self.pop();
                    if !self.is_truthy(condition) {
                        self.current_frame_mut().ip = pos.saturating_sub(1); // -1 because the loop will increment ip
                    }
                }
                OPJUMP => {
                    let pos = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip = pos.saturating_sub(1); // -1 because the loop will increment ip
                }
                OPNULL => {
                    self.push(Object::Null)?;
                }
                OPSETGLOBAL => {
                    let global_index = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip += 2;
                    self.globals.borrow_mut()[global_index] = self.pop();
                }
                OPGETGLOBAL => {
                    let global_index = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip += 2;
                    let value = self.globals.borrow()[global_index].clone();
                    if let Err(err) = self.push(value) {
                        return Err(err);
                    }
                }
                OPSETLOCAL => {
                    // Read u8 for local_index, adjust ip+1 only
                    let local_index = self.current_frame().instructions().0[ip + 1] as usize;
                    self.current_frame_mut().ip += 1;
                    let frame = self.current_frame();
                    let pos = frame.base_pointer + local_index;
                    // Ensure stack has enough space
                    while self.stack.len() <= pos {
                        self.stack.push(Object::Null);
                    }
                    self.stack[pos] = self.pop();
                }
                OPGETLOCAL => {
                    // Read u8 for local_index, adjust ip+1 only
                    let local_index = self.current_frame().instructions().0[ip + 1] as usize;
                    self.current_frame_mut().ip += 1;
                    let frame = self.current_frame();
                    let pos = frame.base_pointer + local_index;
                    let value = self.stack[pos].clone();
                    self.push(value)?;
                }
                OPARRAY => {
                    let num_elements = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip += 2;

                    let array = self.build_array(self.sp - num_elements, self.sp);
                    self.sp = self.sp - num_elements;

                    if let Err(err) = self.push(array) {
                        return Err(err);
                    }
                }
                OPHASH => {
                    let num_elements = read_u16(&self.current_frame().instructions().0[ip + 1..ip + 3]) as usize;
                    self.current_frame_mut().ip += 2;

                    let hash = match self.build_hash(self.sp - num_elements, self.sp) {
                        Ok(h) => h,
                        Err(err) => return Err(err),
                    };
                    self.sp = self.sp - num_elements;

                    if let Err(err) = self.push(hash) {
                        return Err(err);
                    }
                }
                OPINDEX => {
                    let index = self.pop();
                    let left = self.pop();
                    
                    if let Err(err) = self.execute_index_expression(left, index) {
                        return Err(err);
                    }
                }
                OPCALL => {
                    let num_args = self.current_frame().instructions().0[ip + 1] as usize;
                    self.current_frame_mut().ip += 1;

                    let switched_frame = self.execute_call(num_args)?;

                    // Only skip IP increment if we switched frames (compiled function)
                    if switched_frame {
                        continue;
                    }
                    // For builtins, we didn't switch frames, so let IP increment normally
                }
                OPRETURNVALUE => {
                    let return_value = self.pop();

                    let frame = self.pop_frame().expect("no frame to pop");
                    self.sp = frame.base_pointer - 1;

                    let err = self.push(return_value);
                    if let Err(e) = err {
                        return Err(e);
                    }
                }
                OPRETURN => {
                    let frame = self.pop_frame().expect("no frame to pop");
                    self.sp = frame.base_pointer - 1;

                    let err = self.push(Object::Null);
                    if let Err(e) = err {
                        return Err(e);
                    }
                }
                OPBUILTIN => {
                    // Read the builtin index from the next instruction byte
                    let builtin_index = self.current_frame().instructions().0[ip + 1] as usize;
                    self.current_frame_mut().ip += 1;

                    // Get the definition from the builtins list
                    let definition = &BUILTINS[builtin_index];

                    // Push the Builtin object onto the stack
                    if let Err(e) = self.push(Object::Builtin(0, definition.builtin.function.0)) {
                        return Err(e);
                    }
                }
                OPCLOSURE => {
                    let const_index = read_u16(&self.current_frame().instructions().0[ip + 1..]) as usize;
                    let num_free = self.current_frame().instructions().0[ip + 3] as usize;
                    self.current_frame_mut().ip += 3;

                    if let Err(err) = self.push_closure(const_index, num_free) {
                        return Err(err);
                    }
                }
                OPGETFREE => {
                    let free_index = self.current_frame().instructions().0[ip + 1] as usize;
                    self.current_frame_mut().ip += 1;

                    let current_closure = match &self.current_frame().func {
                        Object::Closure(closure) => closure,
                        _ => return Err("OPGETFREE called but current frame is not a closure".to_string()),
                    };

                    if free_index >= current_closure.free.len() {
                        return Err(format!("free index {} out of bounds", free_index));
                    }

                    let obj = current_closure.free[free_index].clone();
                    if let Err(e) = self.push(obj) {
                        return Err(e);
                    }
                }
                OPCURRENTCLOSURE => {
                    let current_closure = match &self.current_frame().func {
                        Object::Closure(closure) => Object::Closure(closure.clone()),
                        _ => return Err("OPCURRENTCLOSURE called but current frame is not a closure".to_string()),
                    };

                    if let Err(e) = self.push(current_closure) {
                        return Err(e);
                    }
                }
                _ => {
                    return Err(format!("unknown opcode: {}", op));
                }
            }
            self.current_frame_mut().ip += 1;
        }
        Ok(())
    }

    fn push(&mut self, o: Object) -> Result<(), String> {
        if self.sp >= STACK_SIZE {
            return Err("stack overflow".to_string());
        }
        self.stack[self.sp] = o;
        self.sp += 1;
        Ok(())
    }

    fn pop(&mut self) -> Object {
        if self.sp == 0 {
            // In production, you might want to return a Result or panic.
            // Here, we just return Null for simplicity.
            return Object::Null;
        }
        self.sp -= 1;
        self.stack[self.sp].clone()
    }

    pub fn last_popped_stack_elem(&self) -> &Object {
        &self.stack[self.sp]
    }

    fn execute_binary_operation(&mut self, op: u8) -> Result<(), String> {
        let right = self.pop();
        let left = self.pop();

        match (&left, &right) {
            (Object::Int(_), Object::Int(_)) => {
                self.execute_binary_integer_operation(op, left, right)
            }
            (Object::String(_), Object::String(_)) => {
                self.execute_binary_string_operation(op, left, right)
            }
            (l, r) => Err(format!(
                "unsupported types for binary operation: {:?} {:?}",
                l, r
            )),
        }
    }

    fn execute_binary_integer_operation(&mut self, op: u8, left: Object, right: Object) -> Result<(), String> {
        let left_value = match left {
            Object::Int(v) => v,
            _ => return Err("left operand is not integer".to_string()),
        };
        let right_value = match right {
            Object::Int(v) => v,
            _ => return Err("right operand is not integer".to_string()),
        };

        let result = match op {
            OPADD => left_value + right_value,
            OPSUB => left_value - right_value,
            OPMULT => left_value * right_value,
            OPDIV => left_value / right_value,
            _ => return Err(format!("unknown integer operator: {}", op)),
        };

        self.push(Object::Int(result))
    }

    fn execute_binary_string_operation(&mut self, op: u8, left: Object, right: Object) -> Result<(), String> {
        if op != OPADD {
            return Err(format!("unknown string operator: {}", op));
        }

        let left_value = match left {
            Object::String(s) => s,
            _ => return Err("left operand is not string".to_string()),
        };
        let right_value = match right {
            Object::String(s) => s,
            _ => return Err("right operand is not string".to_string()),
        };

        self.push(Object::String(left_value + &right_value))
    }

    fn execute_comparison(&mut self, op: u8) -> Result<(), String> {
        let right = self.pop();
        let left = self.pop();

        match (&left, &right) {
            (Object::Int(_), Object::Int(_)) => {
                self.execute_integer_comparison(op, left, right)
            }
            _ => match op {
                OPEQUAL => {
                    self.push(Self::native_bool_to_boolean_object(left == right))
                }
                OPNOTEQUAL => {
                    self.push(Self::native_bool_to_boolean_object(left != right))
                }
                _ => Err(format!(
                    "unknown operator: {} ({:?} {:?})",
                    op, left, right
                )),
            },
        }
    }

    fn execute_integer_comparison(&mut self, op: u8, left: Object, right: Object) -> Result<(), String> {
        let left_value = match left {
            Object::Int(v) => v,
            _ => return Err("left operand is not integer".to_string()),
        };
        let right_value = match right {
            Object::Int(v) => v,
            _ => return Err("right operand is not integer".to_string()),
        };

        let result = match op {
            OPEQUAL => left_value == right_value,
            OPNOTEQUAL => left_value != right_value,
            OPGREATERTHAN => left_value > right_value,
            OPGREATERTHANEQUAL => left_value >= right_value,
            OPLESSTHAN => left_value < right_value,
            OPLESSTHANEQUAL => left_value <= right_value,
            _ => return Err(format!("unknown operator: {}", op)),
        };

        self.push(Self::native_bool_to_boolean_object(result))
    }

    fn execute_bang_operator(&mut self) -> Result<(), String> {
        let operand = self.pop();
        match operand {
            Object::Bool(true) => self.push(Object::Bool(false)),
            Object::Bool(false) => self.push(Object::Bool(true)),
            Object::Null => self.push(Object::Bool(true)),
            _ => self.push(Object::Bool(false)),
        }
    }

    fn execute_minus_operator(&mut self) -> Result<(), String> {
        let operand = self.pop();
        match operand {
            Object::Int(value) => {
                self.push(Object::Int(-value))
            }
            _ => Err(format!(
                "unsupported type for negation: {:?}",
                operand
            )),
        }
    }

    fn execute_plus_operator(&mut self) -> Result<(), String> {
        let operand = self.pop();
        match operand {
            Object::Int(value) => {
                self.push(Object::Int(value))
            }
            _ => Err(format!(
                "unsupported type for plus operator: {:?}",
                operand
            )),
        }
    }

    fn build_array(&self, start: usize, end: usize) -> Object {
        let mut elements = Vec::with_capacity(end - start);
        for i in start..end {
            elements.push(self.stack[i].clone());
        }
        Object::Array(elements)
    }

    fn build_hash(&self, start: usize, end: usize) -> Result<Object, String> {
        let mut pairs = std::collections::HashMap::new();

        let mut i = start;
        while i < end {
            let key = self.stack[i].clone();
            let value = self.stack[i + 1].clone();

            let hash_key = match key.hash_key() {
                Ok(hk) => hk,
                Err(_) => {
                    return Err(format!("unusable as hash key: {:?}", key));
                }
            };

            pairs.insert(hash_key, value);

            i += 2;
        }

        Ok(Object::Hash(pairs))
    }

    fn execute_index_expression(&mut self, left: Object, index: Object) -> Result<(), String> {
        match (&left, &index) {
            (Object::Array(_elements), Object::Int(_idx)) => {
                self.execute_array_index(left, index)
            }
            (Object::Hash(_pairs), _) => {
                self.execute_hash_index(left, index)
            }
            _ => {
                Err(format!("index operator not supported: {:?}", left))
            }
        }
    }

    fn execute_array_index(&mut self, array: Object, index: Object) -> Result<(), String> {
        let array_obj = match array {
            Object::Array(elements) => elements,
            _ => return Err("left object is not an array".to_string()),
        };
        let idx = match index {
            Object::Int(i) => i,
            _ => return self.push(Object::Null),
        };
        let max = array_obj.len() as i64 - 1;
        if idx < 0 || idx > max {
            return self.push(Object::Null);
        }
        self.push(array_obj[idx as usize].clone())
    }

    fn execute_hash_index(&mut self, hash: Object, index: Object) -> Result<(), String> {
        let pairs = match hash {
            Object::Hash(pairs) => pairs,
            _ => return Err("left object is not a hash".to_string()),
        };
        let key = match index.hash_key() {
            Ok(hk) => hk,
            Err(_) => return Err(format!("unusable as hash key: {:?}", index)),
        };
        match pairs.get(&key) {
            Some(value) => self.push(value.clone()),
            None => self.push(Object::Null),
        }
    }

    fn current_frame(&self) -> &Frame {
        &self.frames[self.frames_index - 1]
    }

    fn current_frame_mut(&mut self) -> &mut Frame {
        &mut self.frames[self.frames_index - 1]
    }

    fn call_function(&mut self, num_args: usize) -> Result<(), String> {
        // The function object is located on the stack just before the arguments
        let func_pos = self.sp - 1 - num_args;
        let function = match self.stack.get(func_pos) {
            Some(Object::CompiledFunction(instructions, num_locals, num_params)) => {
                // Check argument count matches function's parameter count
                if num_args != *num_params {
                    return Err(format!(
                        "wrong number of arguments: want={}, got={}",
                        num_params, num_args
                    ));
                }
                // Clone instructions so the frame owns it
                (instructions.clone(), *num_locals, *num_params)
            }
            _ => return Err("calling non-function".to_string()),
        };

        // Create a new frame
        let frame = Frame::new(Object::CompiledFunction(function.0.clone(), function.1, function.2), func_pos + 1);
        self.push_frame(frame.clone());

        // After pushing frame, move sp to make space for local variables
        self.sp = frame.base_pointer + function.1;

        Ok(())
    }

    fn call_builtin(&mut self, builtin_fn: fn(Vec<Object>) -> Object, num_args: usize) -> Result<(), String> {
        // Arguments are at stack[sp - num_args .. sp]
        let args = self.stack[self.sp - num_args .. self.sp].to_vec();

        let result = builtin_fn(args);
        
        // Remove the arguments and the builtin function from the stack
        self.sp = self.sp - num_args - 1;

        // Push the result
        self.push(result)?;

        Ok(())
    }

    fn push_closure(&mut self, const_index: usize, num_free: usize) -> Result<(), String> {
        // Get the constant at const_index and ensure it's a CompiledFunction
        let constant = self.constants.get(const_index)
            .ok_or_else(|| format!("no constant at index {}", const_index))?;

        let function = match constant {
            Object::CompiledFunction(instructions, num_locals, num_params) => {
                crate::object::CompiledFunction {
                    instructions: instructions.clone(),
                    num_locals: *num_locals,
                    num_parameters: *num_params,
                }
            }
            _ => {
                return Err(format!("not a function: {:?}", constant));
            }
        };

        // Gather free variables (they are the last num_free objects on the stack)
        let mut free = Vec::with_capacity(num_free);
        for i in 0..num_free {
            let idx = self.sp - num_free + i;
            free.push(self.stack.get(idx)
                .cloned()
                .unwrap_or(Object::Null));
        }
        // Remove the free variables from the stack
        self.sp -= num_free;

        let closure = crate::object::Closure {
            func: function,
            free,
        };

        self.push(Object::Closure(closure))
    }

    fn push_frame(&mut self, frame: Frame) {
        if self.frames_index < self.frames.len() {
            self.frames[self.frames_index] = frame;
        } else {
            self.frames.push(frame);
        }
        self.frames_index += 1;
    }

    fn pop_frame(&mut self) -> Option<Frame> {
        if self.frames_index == 0 {
            None
        } else {
            self.frames_index -= 1;
            Some(self.frames.remove(self.frames_index))
        }
    }

    fn execute_call(&mut self, num_args: usize) -> Result<bool, String> {
        // The callee is at stack[sp - 1 - num_args]
        let callee_pos = self.sp - 1 - num_args;
        let callee_obj = self.stack.get(callee_pos).cloned().ok_or_else(|| {
            "no function found at call position".to_string()
        })?;

        match callee_obj {
            Object::CompiledFunction(_, _, _) => {
                self.call_function(num_args)?;
                Ok(true) // Switched frames
            }
            Object::Closure(closure) => {
                self.call_closure(closure, num_args)
            }
            Object::Builtin(_, builtin_fn) => {
                self.call_builtin(builtin_fn, num_args)?;
                Ok(false) // Didn't switch frames
            }
            _ => Err("calling non-function and non-built-in".to_string()),
        }
    }

    fn call_closure(&mut self, closure: crate::object::Closure, num_args: usize) -> Result<bool, String> {
        if num_args != closure.func.num_parameters {
            return Err(format!(
                "wrong number of arguments: want={}, got={}",
                closure.func.num_parameters, num_args
            ));
        }

        let frame = Frame::new(Object::Closure(closure), self.sp - num_args);
        self.push_frame(frame);

        // After pushing a new frame, set sp for new frame: base_pointer + num_locals
        let current_frame = self.current_frame();
        if let Object::Closure(ref closure) = current_frame.func {
            self.sp = current_frame.base_pointer + closure.func.num_locals;
        }

        Ok(true)
    }
    
    fn native_bool_to_boolean_object(input: bool) -> Object {
        if input {
            return TRUE
        }
        return FALSE
    }

    fn is_truthy(&self, obj: Object) -> bool {
        match obj {
            Object::Bool(false) => false,
            Object::Null => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler::Compiler;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::object::Object;
    use std::collections::HashMap;

    #[derive(Clone)]
    struct VmTestCase<'a> {
        input: &'a str,
        expected: Object,
    }

    fn parse(input: &str) -> crate::parser::ast::Program {
        let l = Lexer::new(input);
        let mut p = Parser::new(l);
        p.parse()
    }

    fn run_vm_tests(tests: &[VmTestCase]) {
        for tt in tests {
            let program = parse(tt.input);

            let mut compiler = Compiler::new();
            if let Err(e) = compiler.compile(&program) {
                panic!("compiler error: {}", e);
            }

            let bytecode = compiler.bytecode();
            let mut vm = super::VM::new(&bytecode);

            if let Err(e) = vm.run() {
                panic!("vm error: {}", e);
            }

            let stack_elem = vm.last_popped_stack_elem();
            test_expected_object(tt.expected.clone(), stack_elem);
        }
    }

    fn test_expected_object(expected: Object, actual: &Object) {
        match (&expected, actual) {
            (Object::Int(expected_int), Object::Int(actual_int)) => {
                assert_eq!(
                    *actual_int, *expected_int,
                    "object has wrong value. got={}, want={}",
                    actual_int, expected_int
                );
            }
            (Object::Bool(expected_bool), Object::Bool(actual_bool)) => {
                assert_eq!(
                    *actual_bool, *expected_bool,
                    "object has wrong value. got={}, want={}",
                    actual_bool, expected_bool
                );
            }
            (Object::String(expected_string), Object::String(actual_string)) => {
                assert_eq!(
                    *actual_string, *expected_string,
                    "object has wrong value. got={}, want={}",
                    actual_string, expected_string
                );
            }
            (Object::Null, Object::Null) => {
                assert_eq!(
                    actual, &expected,
                    "object has wrong value. got={:?}, want={:?}",
                    actual, expected
                );
            }
            (Object::Array(expected_elems), Object::Array(actual_elems)) => {
                assert_eq!(
                    actual_elems.len(), expected_elems.len(),
                    "wrong num of elements. want={}, got={}",
                    expected_elems.len(), actual_elems.len()
                );
                for (_i, (expected_elem, actual_elem)) in expected_elems.iter().zip(actual_elems.iter()).enumerate() {
                    test_expected_object(expected_elem.clone(), actual_elem);
                }
            }
            (Object::Hash(expected_map), Object::Hash(actual_map)) => {
                assert_eq!(
                    actual_map.len(), expected_map.len(),
                    "hash has wrong number of pairs. want={}, got={}",
                    expected_map.len(), actual_map.len()
                );
                for (expected_key, expected_value) in expected_map {
                    match actual_map.get(expected_key) {
                        Some(actual_value) => {
                            test_expected_object(expected_value.clone(), actual_value);
                        }
                        None => {
                            panic!(
                                "no pair for given key in actual hash. key={:?}",
                                expected_key
                            );
                        }
                    }
                }
            }
            _ => {
                assert_eq!(
                    actual, &expected,
                    "object has wrong value. got={:?}, want={:?}",
                    actual, expected
                );
            }
        }
    }


    #[test]
    fn test_integer_arithmetic() {
        let tests = [
            VmTestCase { input: "1", expected: Object::Int(1) },
            VmTestCase { input: "2", expected: Object::Int(2) },
            VmTestCase { input: "1 + 2", expected: Object::Int(3) },
            VmTestCase { input: "1 - 2", expected: Object::Int(-1) },
            VmTestCase { input: "1 * 2", expected: Object::Int(2) },
            VmTestCase { input: "4 / 2", expected: Object::Int(2) },
            VmTestCase { input: "50 / 2 * 2 + 10 - 5", expected: Object::Int(55) },
            VmTestCase { input: "5 + 5 + 5 + 5 - 10", expected: Object::Int(10) },
            VmTestCase { input: "2 * 2 * 2 * 2 * 2", expected: Object::Int(32) },
            VmTestCase { input: "5 * 2 + 10", expected: Object::Int(20) },
            VmTestCase { input: "5 + 2 * 10", expected: Object::Int(25) },
            VmTestCase { input: "5 * (2 + 10)", expected: Object::Int(60) },
            VmTestCase { input: "-5", expected: Object::Int(-5) },
            VmTestCase { input: "-10", expected: Object::Int(-10) }, 
            VmTestCase { input: "-50 + 100 + -50", expected: Object::Int(0) },
            VmTestCase { input: "(5 + 10 * 2 + 15 / 3) * 2 + -10", expected: Object::Int(50) },
            VmTestCase { input: "+5 - 2", expected: Object::Int(3) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_boolean_expressions() {
        let tests = [
            VmTestCase { input: "true", expected: Object::Bool(true) },
            VmTestCase { input: "false", expected: Object::Bool(false) },
            VmTestCase { input: "1 < 2", expected: Object::Bool(true) },
            VmTestCase { input: "1 > 2", expected: Object::Bool(false) },
            VmTestCase { input: "1 < 1", expected: Object::Bool(false) },
            VmTestCase { input: "1 > 1", expected: Object::Bool(false) },
            VmTestCase { input: "1 == 1", expected: Object::Bool(true) },
            VmTestCase { input: "1 != 1", expected: Object::Bool(false) },
            VmTestCase { input: "1 == 2", expected: Object::Bool(false) },
            VmTestCase { input: "1 != 2", expected: Object::Bool(true) },
            VmTestCase { input: "true == true", expected: Object::Bool(true) },
            VmTestCase { input: "false == false", expected: Object::Bool(true) },
            VmTestCase { input: "true == false", expected: Object::Bool(false) },
            VmTestCase { input: "true != false", expected: Object::Bool(true) },
            VmTestCase { input: "false != true", expected: Object::Bool(true) },
            VmTestCase { input: "(1 < 2) == true", expected: Object::Bool(true) },
            VmTestCase { input: "(1 < 2) == false", expected: Object::Bool(false) },
            VmTestCase { input: "(1 > 2) == false", expected: Object::Bool(true) },
            VmTestCase { input: "(1 > 2) == true", expected: Object::Bool(false) },
            VmTestCase { input: "(1 == 2) == false", expected: Object::Bool(true) },
            VmTestCase { input: "(1 == 2) == true", expected: Object::Bool(false) },
            VmTestCase { input: "!(if (false) { 5; })", expected: Object::Bool(true) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_conditionals() {
        let tests = [
            VmTestCase { input: "if (true) { 10 }", expected: Object::Int(10) },
            VmTestCase { input: "if (true) { 10 } else { 20 }", expected: Object::Int(10) },
            VmTestCase { input: "if (false) { 10 } else { 20 }", expected: Object::Int(20) },
            VmTestCase { input: "if (1) { 10 }", expected: Object::Int(10) },
            VmTestCase { input: "if (1 < 2) { 10 }", expected: Object::Int(10) },
            VmTestCase { input: "if (1 < 2) { 10 } else { 20 }", expected: Object::Int(10) },
            VmTestCase { input: "if (1 > 2) { 10 } else { 20 }", expected: Object::Int(20) },
            VmTestCase { input: "if (1 > 2) { 10 }", expected: Object::Null },
            VmTestCase { input: "if (false) { 10 }", expected: Object::Null },
            VmTestCase { input: "if ((if (false) { 10 })) { 10 } else { 20 }", expected: Object::Int(20) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_global_let_statements() {
        let tests = [
            VmTestCase { input: "let one = 1; one", expected: Object::Int(1) },
            VmTestCase { input: "let one = 1; let two = 2; one + two", expected: Object::Int(3) },
            VmTestCase { input: "let one = 1; let two = one + one; one + two", expected: Object::Int(3) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_string_expressions() {
        let tests = [
            VmTestCase { input: r#""monkey""#, expected: Object::String("monkey".to_string()) },
            VmTestCase { input: r#""mon" + "key""#, expected: Object::String("monkey".to_string()) },
            VmTestCase { input: r#""mon" + "key" + "banana""#, expected: Object::String("monkeybanana".to_string()) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_array_literals() {
        let tests = [
            VmTestCase { input: "[]", expected: Object::Array(vec![]) },
            VmTestCase { input: "[1, 2, 3]", expected: Object::Array(vec![
                Object::Int(1),
                Object::Int(2),
                Object::Int(3),
            ]) },
            VmTestCase { input: "[1 + 2, 3 * 4, 5 + 6]", expected: Object::Array(vec![
                Object::Int(3),
                Object::Int(12),
                Object::Int(11),
            ]) },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_hash_literals() {
        let tests = [
            VmTestCase { input: "{}", expected: Object::Hash(HashMap::new()) },
            VmTestCase { 
                input: "{1: 2, 2: 3}", 
                expected: {
                    let mut m = HashMap::new();
                    m.insert(Object::Int(1), Object::Int(2));
                    m.insert(Object::Int(2), Object::Int(3));
                    Object::Hash(m)
                },
            },
            VmTestCase { 
                input: "{1 + 1: 2 * 2, 3 + 3: 4 * 4}", 
                expected: {
                    let mut m = HashMap::new();
                    m.insert(Object::Int(2), Object::Int(4));
                    m.insert(Object::Int(6), Object::Int(16));
                    Object::Hash(m)
                },
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_index_expressions() {
        let tests = [
            VmTestCase { input: "[1, 2, 3][1]", expected: Object::Int(2) },
            VmTestCase { input: "[1, 2, 3][0 + 2]", expected: Object::Int(3) },
            VmTestCase { input: "[[1, 1, 1]][0][0]", expected: Object::Int(1) },
            VmTestCase { input: "[][0]", expected: Object::Null },
            VmTestCase { input: "[1, 2, 3][99]", expected: Object::Null },
            VmTestCase { input: "[1][-1]", expected: Object::Null },
            VmTestCase { 
                input: "{1: 1, 2: 2}[1]", 
                expected: Object::Int(1),
            },
            VmTestCase { 
                input: "{1: 1, 2: 2}[2]", 
                expected: Object::Int(2),
            },
            VmTestCase { 
                input: "{1: 1}[0]", 
                expected: Object::Null,
            },
            VmTestCase { 
                input: "{}[0]", 
                expected: Object::Null,
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_calling_functions_without_arguments() {
        let tests = [
            VmTestCase {
                input: r#"
                    let fivePlusTen = fn() { 5 + 10; };
                    fivePlusTen();
                "#,
                expected: Object::Int(15),
            },
            VmTestCase {
                input: r#"
                    let one = fn() { 1; };
                    let two = fn() { 2; };
                    one() + two()
                "#,
                expected: Object::Int(3),
            },
            VmTestCase {
                input: r#"
                    let a = fn() { 1 };
                    let b = fn() { a() + 1 };
                    let c = fn() { b() + 1 };
                    c();
                "#,
                expected: Object::Int(3),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_calling_functions_with_bindings() {
        let tests = [
            VmTestCase {
                input: r#"
                    let one = fn() { let one = 1; one };
                    one();
                "#,
                expected: Object::Int(1),
            },
            VmTestCase {
                input: r#"
                    let oneAndTwo = fn() { let one = 1; let two = 2; one + two; };
                    oneAndTwo();
                "#,
                expected: Object::Int(3),
            },
            VmTestCase {
                input: r#"
                    let oneAndTwo = fn() { let one = 1; let two = 2; one + two; };
                    let threeAndFour = fn() { let three = 3; let four = 4; three + four; };
                    oneAndTwo() + threeAndFour();
                "#,
                expected: Object::Int(10),
            },
            VmTestCase {
                input: r#"
                    let firstFoobar = fn() { let foobar = 50; foobar; };
                    let secondFoobar = fn() { let foobar = 100; foobar; };
                    firstFoobar() + secondFoobar();
                "#,
                expected: Object::Int(150),
            },
            VmTestCase {
                input: r#"
                    let globalSeed = 50;
                    let minusOne = fn() {
                        let num = 1;
                        globalSeed - num;
                    }
                    let minusTwo = fn() {
                        let num = 2;
                        globalSeed - num;
                    }
                    minusOne() + minusTwo();
                "#,
                expected: Object::Int(97),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_first_class_functions() {
        let tests = [
            VmTestCase {
                input: r#"
                    let returnsOne = fn() { 1; };
                    let returnsOneReturner = fn() { returnsOne; };
                    returnsOneReturner()();
                "#,
                expected: Object::Int(1),
            },
            VmTestCase {
                input: r#"
                    let returnsOneReturner = fn() {
                        let returnsOne = fn() { 1; };
                        returnsOne;
                    };
                    returnsOneReturner()();
                "#,
                expected: Object::Int(1),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_functions_with_return_statement() {
        let tests = [
            VmTestCase {
                input: r#"
                    let earlyExit = fn() { return 99; 100; };
                    earlyExit();
                "#,
                expected: Object::Int(99),
            },
            VmTestCase {
                input: r#"
                    let earlyExit = fn() { return 99; return 100; };
                    earlyExit();
                "#,
                expected: Object::Int(99),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_functions_without_return_value() {
        let tests = [
            VmTestCase {
                input: r#"
                    let noReturn = fn() { };
                    noReturn();
                "#,
                expected: Object::Null,
            },
            VmTestCase {
                input: r#"
                    let noReturn = fn() { };
                    let noReturnTwo = fn() { noReturn(); };
                    noReturn();
                    noReturnTwo();
                "#,
                expected: Object::Null,
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_function_calls() {
        let tests = [
            VmTestCase {
                input: r#"
                    fn() { 24 }();
                "#,
                expected: Object::Int(24),
            },
            VmTestCase {
                input: r#"
                    let noArg = fn() { 24 };
                    noArg();
                "#,
                expected: Object::Int(24),
            },
            VmTestCase {
                input: r#"
                    let oneArg = fn(a) { a };
                    oneArg(24);
                "#,
                expected: Object::Int(24),
            },
            VmTestCase {
                input: r#"
                    let manyArg = fn(a, b, c) { a; b; c };
                    manyArg(24, 25, 26);
                "#,
                expected: Object::Int(26),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_calling_functions_with_arguments_and_bindings() {
        let tests = [
            VmTestCase {
                input: r#"
                    let identity = fn(a) { a; };
                    identity(4);
                "#,
                expected: Object::Int(4),
            },
            VmTestCase {
                input: r#"
                    let sum = fn(a, b) { a + b; };
                    sum(1, 2);
                "#,
                expected: Object::Int(3),
            },
            VmTestCase {
                input: r#"
                    let sum = fn(a, b) {
                        let c = a + b;
                        c;
                    };
                    sum(1, 2);
                "#,
                expected: Object::Int(3),
            },
            VmTestCase {
                input: r#"
                    let sum = fn(a, b) {
                        let c = a + b;
                        c;
                    };
                    sum(1, 2) + sum(3, 4);
                "#,
                expected: Object::Int(10),
            },
            VmTestCase {
                input: r#"
                    let sum = fn(a, b) {
                        let c = a + b;
                        c;
                    };
                    let outer = fn() {
                        sum(1, 2) + sum(3, 4);
                    };
                    outer();
                "#,
                expected: Object::Int(10),
            },
            VmTestCase {
                input: r#"
                    let globalNum = 10;

                    let sum = fn(a, b) {
                        let c = a + b;
                        c + globalNum;
                    };

                    let outer = fn() {
                        sum(1, 2) + sum(3, 4) + globalNum;
                    };

                    outer() + globalNum;
                "#,
                expected: Object::Int(50),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_calling_functions_with_wrong_arguments() {
        let tests = [
            VmTestCase {
                input: r#"fn() { 1; }(1);"#,
                expected: Object::String("wrong number of arguments: want=0, got=1".to_string()),
            },
            VmTestCase {
                input: r#"fn(a) { a; }();"#,
                expected: Object::String("wrong number of arguments: want=1, got=0".to_string()),
            },
            VmTestCase {
                input: r#"fn(a, b) { a + b; }(1);"#,
                expected: Object::String("wrong number of arguments: want=2, got=1".to_string()),
            },
        ];

        for tt in tests.iter() {
            let program = parse(tt.input);

            let mut compiler = Compiler::new();
            if let Err(e) = compiler.compile(&program) {
                panic!("compiler error: {}", e);
            }

            let bytecode = compiler.bytecode();
            let mut vm = super::VM::new(&bytecode);

            let res = vm.run();
            match &tt.expected {
                Object::String(expected_error) => {
                    match res {
                        Ok(_) => panic!("expected VM error but resulted in none."),
                        Err(got) => {
                            assert_eq!(
                                expected_error, &got,
                                "wrong VM error: want={:?}, got={:?}", expected_error, got
                            );
                        }
                    }
                }
                expected => {
                    if let Err(e) = res {
                        panic!("unexpected vm error: {}", e);
                    }
                    let stack_elem = vm.last_popped_stack_elem();
                    test_expected_object(expected.clone(), stack_elem);
                }
            }
        }
    }

    #[test]
    fn test_builtin_functions() {
        use crate::object::Object;

        #[derive(Clone)]
        enum Expected {
            Int(i64),
            Null,
            String(String),
            Array(Vec<Object>),
        }

        struct BuiltinVmTestCase<'a> {
            input: &'a str,
            expected: Expected,
        }

        // Helper to allow brevity in test definition
        fn err(msg: &str) -> Expected {
            Expected::String(msg.to_string())
        }

        let tests = [
            BuiltinVmTestCase {
                input: r#"len("")"#,
                expected: Expected::Int(0),
            },
            BuiltinVmTestCase {
                input: r#"len("four")"#,
                expected: Expected::Int(4),
            },
            BuiltinVmTestCase {
                input: r#"len("hello world")"#,
                expected: Expected::Int(11),
            },
            BuiltinVmTestCase {
                input: r#"len(1)"#,
                expected: err("argument to `len` not supported, got INTEGER"),
            },
            BuiltinVmTestCase {
                input: r#"len("one", "two")"#,
                expected: err("wrong number of arguments. got=2, want=1"),
            },
            BuiltinVmTestCase {
                input: r#"len([1, 2, 3])"#,
                expected: Expected::Int(3),
            },
            BuiltinVmTestCase {
                input: r#"len([])"#,
                expected: Expected::Int(0),
            },
            BuiltinVmTestCase {
                input: r#"puts("hello", "world!")"#,
                expected: Expected::Null,
            },
            BuiltinVmTestCase {
                input: r#"first([1, 2, 3])"#,
                expected: Expected::Int(1),
            },
            BuiltinVmTestCase {
                input: r#"first([])"#,
                expected: Expected::Null,
            },
            BuiltinVmTestCase {
                input: r#"first(1)"#,
                expected: err("argument to `first` must be ARRAY, got INTEGER"),
            },
            BuiltinVmTestCase {
                input: r#"last([1, 2, 3])"#,
                expected: Expected::Int(3),
            },
            BuiltinVmTestCase {
                input: r#"last([])"#,
                expected: Expected::Null,
            },
            BuiltinVmTestCase {
                input: r#"last(1)"#,
                expected: err("argument to `last` must be ARRAY, got INTEGER"),
            },
            BuiltinVmTestCase {
                input: r#"rest([1, 2, 3])"#,
                expected: Expected::Array(vec![Object::Int(2), Object::Int(3)]),
            },
            BuiltinVmTestCase {
                input: r#"rest([])"#,
                expected: Expected::Null,
            },
            BuiltinVmTestCase {
                input: r#"push([], 1)"#,
                expected: Expected::Array(vec![Object::Int(1)]),
            },
            BuiltinVmTestCase {
                input: r#"push(1, 1)"#,
                expected: err("argument to `push` must be ARRAY, got INTEGER"),
            },
        ];

        for tt in tests.iter() {
            let program = parse(tt.input);

            let mut compiler = Compiler::new();
            if let Err(e) = compiler.compile(&program) {
                panic!("compiler error: {}", e);
            }

            let bytecode = compiler.bytecode();
            let mut vm = super::VM::new(&bytecode);

            let res = vm.run();

            match &tt.expected {
                Expected::String(expected_error) => {
                    // Builtin errors are Object::Error on the stack, not VM errors
                    if let Err(e) = res {
                        panic!("unexpected vm error: {}", e);
                    }
                    let stack_elem = vm.last_popped_stack_elem();
                    match stack_elem {
                        Object::Error(got) => assert_eq!(expected_error, got, "wrong error: want={:?}, got={:?}", expected_error, got),
                        other => panic!("expected error object, got {:?}", other),
                    }
                }
                Expected::Int(expected_value) => {
                    if let Err(e) = res {
                        panic!("unexpected vm error: {}", e);
                    }
                    let stack_elem = vm.last_popped_stack_elem();
                    match stack_elem {
                        Object::Int(got) => assert_eq!(*expected_value, *got, "wrong int result: want={}, got={}", expected_value, got),
                        other => panic!("expected int object, got {:?}", other),
                    }
                }
                Expected::Null => {
                    if let Err(e) = res {
                        panic!("unexpected vm error: {}", e);
                    }
                    let stack_elem = vm.last_popped_stack_elem();
                    assert_eq!(&Object::Null, stack_elem, "expected Null, got {:?}", stack_elem);
                }
                Expected::Array(expected_vec) => {
                    if let Err(e) = res {
                        panic!("unexpected vm error: {}", e);
                    }
                    let stack_elem = vm.last_popped_stack_elem();
                    match stack_elem {
                        Object::Array(got_arr) => assert_eq!(&expected_vec[..], &got_arr[..], "expected array {:?}, got {:?}", expected_vec, got_arr),
                        other => panic!("expected array object, got {:?}", other),
                    }
                }
            }
        }
    }

    #[test]
    fn test_closures() {
        let tests = [
            VmTestCase {
                input: r#"
                    let newClosure = fn(a) {
                        fn() { a; };
                    };
                    let closure = newClosure(99);
                    closure();
                "#,
                expected: Object::Int(99),
            },
            VmTestCase {
                input: r#"
                    let newAdder = fn(a, b) {
                        fn(c) { a + b + c };
                    };
                    let adder = newAdder(1, 2);
                    adder(8);
                "#,
                expected: Object::Int(11),
            },
            VmTestCase {
                input: r#"
                    let newAdder = fn(a, b) {
                        let c = a + b;
                        fn(d) { c + d };
                    };
                    let adder = newAdder(1, 2);
                    adder(8);
                "#,
                expected: Object::Int(11),
            },
            VmTestCase {
                input: r#"
                    let newAdderOuter = fn(a, b) {
                        let c = a + b;
                        fn(d) {
                            let e = d + c;
                            fn(f) { e + f; };
                        };
                    };
                    let newAdderInner = newAdderOuter(1, 2);
                    let adder = newAdderInner(3);
                    adder(8);
                "#,
                expected: Object::Int(14),
            },
            VmTestCase {
                input: r#"
                    let a = 1;
                    let newAdderOuter = fn(b) {
                        fn(c) {
                            fn(d) { a + b + c + d };
                        };
                    };
                    let newAdderInner = newAdderOuter(2);
                    let adder = newAdderInner(3);
                    adder(8);
                "#,
                expected: Object::Int(14),
            },
            VmTestCase {
                input: r#"
                    let newClosure = fn(a, b) {
                        let one = fn() { a; };
                        let two = fn() { b; };
                        fn() { one() + two(); };
                    };
                    let closure = newClosure(9, 90);
                    closure();
                "#,
                expected: Object::Int(99),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_recursive_functions() {
        let tests = [
            VmTestCase {
                input: r#"
                    let countDown = fn(x) {
                        if (x == 0) {
                            return 0;
                        } else {
                            countDown(x - 1);
                        }
                    };
                    countDown(1);
                "#,
                expected: Object::Int(0),
            },
            VmTestCase {
                input: r#"
                    let countDown = fn(x) {
                        if (x == 0) {
                            return 0;
                        } else {
                            countDown(x - 1);
                        }
                    };
                    let wrapper = fn() {
                        countDown(1);
                    };
                    wrapper();
                "#,
                expected: Object::Int(0),
            },
            VmTestCase {
                input: r#"
                    let wrapper = fn() {
                        let countDown = fn(x) {
                            if (x == 0) {
                                return 0;
                            } else {
                                countDown(x - 1);
                            }
                        };
                        countDown(1);
                    };
                    wrapper();
                "#,
                expected: Object::Int(0),
            },
        ];

        run_vm_tests(&tests);
    }

    #[test]
    fn test_recursive_fibonacci() {
        let tests = vec![
            VmTestCase {
                input: r#"
                    let fibonacci = fn(x) {
                        if (x == 0) {
                            return 0;
                        } else {
                            if (x == 1) {
                                return 1;
                            } else {
                                fibonacci(x - 1) + fibonacci(x - 2);
                            }
                        }
                    };
                    fibonacci(15);
                "#,
                expected: Object::Int(610),
            },
        ];

        run_vm_tests(&tests);
    }
}
