use std::rc::Rc;
use std::cell::RefCell;
use crate::code::{
    Instructions, 
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
    OPSETGLOBAL,
    OPGETGLOBAL,
    read_u16, 
};
use crate::compiler::Bytecode;
use crate::evaluator::object::Object;

pub const STACK_SIZE: usize = 2048;
pub const GLOBAL_SIZE: usize = 65536;

pub const TRUE: Object = Object::Bool(true);
pub const FALSE: Object = Object::Bool(false);


pub struct VM {
    pub constants: Vec<Object>,
    pub instructions: Instructions,
    pub globals: Rc<RefCell<Vec<Object>>>,

    stack: Vec<Object>,
    sp: usize, // Always points to the next value. Top of stack is stack[sp-1]
}

impl VM {
    pub fn new(bytecode: &Bytecode) -> Self {
        VM {
            instructions: bytecode.instructions.clone(),
            constants: bytecode.constants.clone(),
            globals: Rc::new(RefCell::new(vec![Object::Null; GLOBAL_SIZE])),
            stack: vec![Object::Null; STACK_SIZE],
            sp: 0,
        }
    }

    pub fn new_with_globals_store(
        bytecode: Bytecode,
        globals: Rc<RefCell<Vec<Object>>>,
    ) -> Self {
        let mut stack = Vec::with_capacity(STACK_SIZE);
        // Pre-fill the stack so that we can easily put values with stack pointer.
        for _ in 0..STACK_SIZE {
            stack.push(Object::Null);
        }

        VM {
            instructions: bytecode.instructions.clone(),
            constants: bytecode.constants,
            globals,
            stack,
            sp: 0,
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
        let mut ip = 0;
        while ip < self.instructions.len() {
            let op = self.instructions.get(ip).unwrap_or(0);
            match op {
                OPCONSTANT => {
                    let const_index = read_u16(&self.instructions.0[ip + 1..ip + 3]) as usize;
                    ip += 2;
                    let obj = self.constants.get(const_index)
                        .ok_or_else(|| format!("constant at {} not found", const_index))?
                        .clone();
                    self.push(obj)?;
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
                    let pos = read_u16(&self.instructions.0[ip + 1..ip + 3]) as usize;
                    ip += 2;
                    let condition = self.pop();
                    if !self.is_truthy(condition) {
                        ip = pos - 1; // -1 because the loop will increment ip
                    }
                }
                OPJUMP => {
                    let pos = read_u16(&self.instructions.0[ip + 1..ip + 3]) as usize;
                    ip = pos - 1; // -1 because the loop will increment ip
                }
                OPNULL => {
                    self.push(Object::Null)?;
                }
                OPSETGLOBAL => {
                    let global_index = read_u16(&self.instructions.0[ip + 1..ip + 3]) as usize;
                    ip += 2;
                    self.globals.borrow_mut()[global_index] = self.pop();
                }
                OPGETGLOBAL => {
                    let global_index = read_u16(&self.instructions.0[ip + 1..ip + 3]) as usize;
                    ip += 2;
                    let value = self.globals.borrow()[global_index].clone();
                    let _ = self.push(value);
                }
                _ => {
                    return Err(format!("unknown opcode: {}", op));
                }
            }
            ip += 1;
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
    use crate::evaluator::object::Object;

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
            (Object::Null, Object::Null) => {
                assert_eq!(
                    actual, &expected,
                    "object has wrong value. got={:?}, want={:?}",
                    actual, expected
                );
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
        use crate::evaluator::object::Object;

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
}
