pub mod symbol_table;
use symbol_table::SymbolTable;
use std::rc::Rc;
use std::cell::RefCell;

use crate::parser::ast;
use crate::code::{
    self, 
    Instructions, 
    Opcode, 
    OPCONSTANT, 
    OPTRUE, 
    OPFALSE, 
    OPADD, 
    OPSUB, 
    OPMULT, 
    OPDIV, 
    OPLESSTHAN, 
    OPLESSTHANEQUAL, 
    OPGREATERTHAN, 
    OPGREATERTHANEQUAL,
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
    OPSETGLOBAL
};
use crate::evaluator::object::Object;

pub struct Compiler {
    pub instructions: Instructions,
    pub constants: Rc<RefCell<Vec<Object>>>,
    last_instruction: EmittedInstruction,
    previous_instruction: EmittedInstruction,
    symbol_table: Rc<RefCell<SymbolTable>>,
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            instructions: Instructions::new(),
            constants: Rc::new(RefCell::new(vec![])),
            last_instruction: EmittedInstruction::new(),
            previous_instruction: EmittedInstruction::new(),
            symbol_table: Rc::new(RefCell::new(SymbolTable::new())),
        }
    }

    pub fn new_with_state(symbol_table: Rc<RefCell<SymbolTable>>, constants: Rc<RefCell<Vec<Object>>>) -> Self {
        Compiler {
            instructions: Instructions::new(),
            constants: constants,
            last_instruction: EmittedInstruction::new(),
            previous_instruction: EmittedInstruction::new(),
            symbol_table: symbol_table,
        }
    }

    pub fn compile(&mut self, program: &ast::Program) -> Result<(), String> {
        for stmt in program {
            self.compile_stmt(stmt)?;
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &ast::Stmt) -> Result<(), String> {
        match stmt {
            ast::Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                self.emit(OPPOP, &[]);
            }
            ast::Stmt::Let(ident, expr) => {
                self.compile_expr(expr)?;
                let symbol = self.symbol_table.borrow_mut().define(&ident.0);
                self.emit(OPSETGLOBAL, &[symbol.index as i32]);
            }
            _ => {
                return Err(format!("unsupported statement: {:?}", stmt));
            }
        }
        Ok(())
    }

    fn compile_expr(&mut self, expr: &ast::Expr) -> Result<(), String> {
        match expr {
            ast::Expr::Lit(literal) => {
                match literal {
                    ast::Literal::Int(value) => {
                        let integer = Object::Int(*value);
                        let const_index = self.add_constant(integer) as i32;
                        self.emit(OPCONSTANT, &[const_index]);
                    }
                    ast::Literal::Bool(value) => {
                        if *value {
                            self.emit(OPTRUE, &[]);
                        } else {
                            self.emit(OPFALSE, &[]);
                        }
                    }
                    _ => {
                        return Err(format!("unsupported literal: {:?}", literal));
                    }
                }
            }
            ast::Expr::Infix(op, left, right) => {
                self.compile_expr(left)?;
                self.compile_expr(right)?;
                match op {
                    ast::Infix::Plus => {
                        self.emit(OPADD, &[]);
                    }
                    ast::Infix::Minus => {
                        self.emit(OPSUB, &[]);
                    }
                    ast::Infix::Multiply => {
                        self.emit(OPMULT, &[]);
                    }
                    ast::Infix::Divide => {
                        self.emit(OPDIV, &[]);
                    }
                    ast::Infix::GreaterThan => {
                        self.emit(OPGREATERTHAN, &[]);
                    }
                    ast::Infix::GreaterThanEqual => {
                        self.emit(OPGREATERTHANEQUAL, &[]);
                    }
                    ast::Infix::LessThan => {
                        self.emit(OPLESSTHAN, &[]);
                    }
                    ast::Infix::LessThanEqual => {
                        self.emit(OPLESSTHANEQUAL, &[]);
                    }
                    ast::Infix::Equal => {
                        self.emit(OPEQUAL, &[]);
                    }
                    ast::Infix::NotEqual => {
                        self.emit(OPNOTEQUAL, &[]);
                    }
                }
            }
            ast::Expr::Prefix(prefix, right) => {
                self.compile_expr(right)?;
                match prefix {
                    ast::Prefix::Not => {
                        self.emit(OPBANG, &[]);
                    }
                    ast::Prefix::PreMinus => {
                        self.emit(OPMINUS, &[]);
                    }
                    ast::Prefix::PrePlus => {
                        self.emit(OPPLUS, &[]);
                    }
                }
            }
            ast::Expr::If { cond, consequence, alternative } => {
                self.compile_expr(cond)?;
                let else_jump = self.emit(OPJUMPELSE, &[9999]);
                
                for stmt in consequence {
                    self.compile_stmt(stmt)?;
                }

                if self.last_instruction_is_pop() {
                    self.remove_last_pop();
                }

                let jump_pos = self.emit(OPJUMP, &[9999]);
                
                let after_consequence_pos = self.instructions.len();
                
                self.change_operand(else_jump, after_consequence_pos as i32);
                
                if let Some(alternative) = alternative {
                    for stmt in alternative {
                        self.compile_stmt(stmt)?;
                    }
                    if self.last_instruction_is_pop() {
                        self.remove_last_pop();
                    }
                } else {
                    self.emit(OPNULL, &[]);
                }

                let after_alternative_pos = self.instructions.len();
                self.change_operand(jump_pos, after_alternative_pos as i32);
            }
            ast::Expr::Ident(ident) => {
                let symbol_index = {
                    let symbol_table = self.symbol_table.borrow();
                    let symbol = symbol_table.resolve(&ident.0);
                    if let Some(symbol) = symbol {
                        Some(symbol.index)
                    } else {
                        None
                    }
                };
                if let Some(index) = symbol_index {
                    self.emit(OPGETGLOBAL, &[index as i32]);
                } else {
                    return Err(format!("undefined variable: {}", ident.0));
                }
            }
            _ => {
                return Err(format!("unsupported expression: {:?}", expr));
            }
        }
        Ok(())
    }

    pub fn bytecode(&self) -> Bytecode {
        Bytecode {
            instructions: self.instructions.clone(),
            constants: self.constants.borrow().clone(),
        }
    }

    pub fn add_constant(&mut self, obj: Object) -> usize {
        self.constants.borrow_mut().push(obj);
        self.constants.borrow().len() - 1
    }

    pub fn emit(&mut self, op: Opcode, operands: &[i32]) -> usize {
        let ins = code::make(op, operands);
        let pos = self.add_instruction(&ins);

        self.set_last_instruction(op, pos);

        pos
    }

    pub fn set_last_instruction(&mut self, op: Opcode, pos: usize) {
        let previous = self.last_instruction.clone();
        let last = EmittedInstruction { opcode: op, position: pos };

        self.previous_instruction = previous;
        self.last_instruction = last;
    }

    pub fn add_instruction(&mut self, ins: &Instructions) -> usize {
        let pos_new_instruction = self.instructions.len();
        self.instructions.0.extend_from_slice(&ins.0);
        pos_new_instruction
    }

    pub fn last_instruction_is_pop(&self) -> bool {
        self.last_instruction.opcode == OPPOP
    }

    pub fn remove_last_pop(&mut self) {
        self.instructions.0.truncate(self.last_instruction.position);
        self.last_instruction = self.previous_instruction.clone();
    }

    pub fn replace_instruction(&mut self, pos: usize, new_instruction: &Instructions) {
        for (i, byte) in new_instruction.0.iter().enumerate() {
            self.instructions.0[pos + i] = *byte;
        }
    }

    pub fn change_operand(&mut self, op_pos: usize, operand: i32) {
        let op = self.instructions.0[op_pos];
        let opcode = op;
        let new_instruction = code::make(opcode, &[operand]);
        self.replace_instruction(op_pos, &new_instruction);
    }
}

pub struct Bytecode {
    pub instructions: Instructions,
    pub constants: Vec<Object>,
}

#[derive(Clone)]
struct EmittedInstruction {
    opcode: Opcode,
    position: usize,
}

impl EmittedInstruction {
    fn new() -> Self {
        EmittedInstruction {
            opcode: 0,
            position: 0,
        }
    }
}


#[cfg(test)]
mod tests {
    use crate::parser::ast;
    use crate::lexer;
    use crate::parser;
    use crate::evaluator::object::Object;
    use crate::code::{
        self, 
        Instructions, 
        OPCONSTANT, 
        OPTRUE, 
        OPFALSE, 
        OPADD, 
        OPSUB, 
        OPMULT, 
        OPDIV, 
        OPLESSTHAN,
        OPLESSTHANEQUAL, 
        OPGREATERTHAN, 
        OPGREATERTHANEQUAL, 
        OPEQUAL, 
        OPNOTEQUAL, 
        OPMINUS, 
        OPPLUS, 
        OPBANG, 
        OPPOP,
        OPGETGLOBAL,
        OPSETGLOBAL
    };

    struct CompilerTestCase<'a> {
        input: &'a str,
        expected_constants: Vec<i64>,
        expected_instructions: Vec<Instructions>,
    }

    #[test]
    fn test_integer_arithmetic() {
        let tests = vec![
            CompilerTestCase {
                input: "1 + 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPADD, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1; 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPPOP, &[]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 - 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPSUB, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 * 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPMULT, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 / 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPDIV, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "-1",
                expected_constants: vec![1],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPMINUS, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "+1",
                expected_constants: vec![1],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPPLUS, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_boolean_expressions() {
        let tests = vec![
            CompilerTestCase {
                input: "true",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "false",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPFALSE, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 > 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPGREATERTHAN, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 >= 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPGREATERTHANEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 < 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPLESSTHAN, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 <= 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPLESSTHANEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 == 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 != 2",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPNOTEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "true == false",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),
                    code::make(OPFALSE, &[]),
                    code::make(OPEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "true != false",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),
                    code::make(OPFALSE, &[]),
                    code::make(OPNOTEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "!true",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),
                    code::make(OPBANG, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_global_let_statements() {
        let tests = vec![
            CompilerTestCase {
                input: "
                let one = 1;
                let two = 2;
                ",
                expected_constants: vec![1, 2],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPSETGLOBAL, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPSETGLOBAL, &[1]),
                ],
            },
            CompilerTestCase {
                input: "
                let one = 1;
                one;
                ",
                expected_constants: vec![1],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPSETGLOBAL, &[0]),
                    code::make(OPGETGLOBAL, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "
                let one = 1;
                let two = one;
                two;
                ",
                expected_constants: vec![1],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPSETGLOBAL, &[0]),
                    code::make(OPGETGLOBAL, &[0]),
                    code::make(OPSETGLOBAL, &[1]),
                    code::make(OPGETGLOBAL, &[1]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    fn run_compiler_tests(tests: Vec<CompilerTestCase>) {
        for tt in tests {
            let program = parse(tt.input);

            let mut compiler = super::Compiler::new();
            if let Err(e) = compiler.compile(&program) {
                panic!("compiler error: {}", e);
            }

            let bytecode = compiler.bytecode();

            if let Err(e) = test_instructions(&tt.expected_instructions, &bytecode.instructions) {
                panic!("testInstructions failed: {}", e);
            }

            if let Err(e) = test_constants(&tt.expected_constants, &bytecode.constants) {
                panic!("testConstants failed: {}", e);
            }
        }
    }

    fn parse(input: &str) -> ast::Program {
        let l = lexer::Lexer::new(input);
        let mut p = parser::Parser::new(l);
        p.parse()
    }

    fn test_instructions(
        expected: &[Instructions],
        actual: &Instructions,
    ) -> Result<(), String> {
        let concatted = concat_instructions(expected);

        if actual.len() != concatted.len() {
            return Err(format!(
                "wrong instructions length.\nwant={:?}\ngot ={:?}",
                concatted, actual
            ));
        }

        for (i, (got, want)) in actual.iter().zip(concatted.iter()).enumerate() {
            if got != want {
                return Err(format!(
                    "wrong instruction at {}.\nwant={:?}\ngot ={:?}",
                    i, concatted, actual
                ));
            }
        }

        Ok(())
    }

    fn concat_instructions(s: &[Instructions]) -> Vec<u8> {
        let mut out = Vec::new();
        for ins in s {
            out.extend(ins.iter());
        }
        out
    }

    fn test_constants(
        expected: &[i64],
        actual: &[Object],
    ) -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "wrong number of constants. got={}, want={}",
                actual.len(),
                expected.len()
            ));
        }

        for (i, constant) in expected.iter().enumerate() {
            match &actual[i] {
                Object::Int(obj_int) => {
                    if *obj_int != *constant {
                        return Err(format!(
                            "constant {} - object has wrong value. got={}, want={}",
                            i, obj_int, constant
                        ));
                    }
                }
                obj => {
                    return Err(format!(
                        "constant {} - object is not Integer. got={:?}",
                        i, obj
                    ));
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_conditionals() {
        use crate::code::{self, Instructions, OPCONSTANT, OPTRUE, OPJUMPELSE, OPJUMP, OPNULL, OPPOP};

        struct CompilerTestCase<'a> {
            input: &'a str,
            expected_constants: Vec<i64>,
            expected_instructions: Vec<Instructions>,
        }

        let tests = [
            CompilerTestCase {
                input: "
                if (true) { 10 }; 3333;
                ",
                expected_constants: vec![10, 3333],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),                // 0000
                    code::make(OPJUMPELSE, &[10]),          // 0001
                    code::make(OPCONSTANT, &[0]),           // 0004
                    code::make(OPJUMP, &[11]),              // 0007
                    code::make(OPNULL, &[]),                // 0010
                    code::make(OPPOP, &[]),                 // 0011
                    code::make(OPCONSTANT, &[1]),           // 0012
                    code::make(OPPOP, &[]),                 // 0015
                ],
            },
            CompilerTestCase {
                input: "
                if (true) { 10 } else { 20 }; 3333;
                ",
                expected_constants: vec![10, 20, 3333],
                expected_instructions: vec![
                    code::make(OPTRUE, &[]),                // 0000
                    code::make(OPJUMPELSE, &[10]),          // 0001
                    code::make(OPCONSTANT, &[0]),           // 0004
                    code::make(OPJUMP, &[13]),              // 0007
                    code::make(OPCONSTANT, &[1]),           // 0010
                    code::make(OPPOP, &[]),                 // 0013
                    code::make(OPCONSTANT, &[2]),           // 0014
                    code::make(OPPOP, &[]),                 // 0017
                ],
            },
        ];

        run_compiler_tests(tests.iter().map(|t| crate::compiler::tests::CompilerTestCase {
            input: t.input,
            expected_constants: t.expected_constants.clone(),
            expected_instructions: t.expected_instructions.clone(),
        }).collect());
    }
}
