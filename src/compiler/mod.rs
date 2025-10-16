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
    OPSETGLOBAL,
    OPARRAY,
    OPHASH,
    OPINDEX,
    OPCALL,
    OPRETURNVALUE,
    OPRETURN,
    OPGETLOCAL,
    OPSETLOCAL,
};
use crate::evaluator::object::Object;

pub struct CompilationScope {
    pub instructions: Instructions,
    last_instruction: EmittedInstruction,
    previous_instruction: EmittedInstruction,
}

pub struct Compiler {
    pub constants: Rc<RefCell<Vec<Object>>>,
    symbol_table: Rc<RefCell<SymbolTable>>,
    scopes: Vec<CompilationScope>,
    scope_index: usize,
}

impl Compiler {
    pub fn new() -> Self {
        let main_scope = CompilationScope {
            instructions: Instructions::new(),
            last_instruction: EmittedInstruction::new(),
            previous_instruction: EmittedInstruction::new(),
        };

        Compiler {
            constants: Rc::new(RefCell::new(vec![])),
            symbol_table: Rc::new(RefCell::new(SymbolTable::new())),
            scopes: vec![main_scope],
            scope_index: 0,
        }
    }

    pub fn new_with_state(symbol_table: Rc<RefCell<SymbolTable>>, constants: Rc<RefCell<Vec<Object>>>) -> Self {
        let main_scope = CompilationScope {
            instructions: Instructions::new(),
            last_instruction: EmittedInstruction::new(),
            previous_instruction: EmittedInstruction::new(),
        };

        Compiler {
            constants: constants,
            symbol_table: symbol_table,
            scopes: vec![main_scope],
            scope_index: 0,
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
                if symbol.scope == crate::compiler::symbol_table::SymbolScope::Global {
                    self.emit(OPSETGLOBAL, &[symbol.index as i32]);
                } else {
                    self.emit(OPSETLOCAL, &[symbol.index as i32]);
                }
            }
            ast::Stmt::Return(value) => {
                self.compile_expr(value)?;
                self.emit(OPRETURNVALUE, &[]);
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
                    ast::Literal::String(value) => {
                        let string_obj = Object::String(value.clone());
                        let const_index = self.add_constant(string_obj) as i32;
                        self.emit(OPCONSTANT, &[const_index]);
                    }
                    ast::Literal::Array(value) => {
                        for elem in value {
                            self.compile_expr(elem)?;
                        }
                        self.emit(OPARRAY, &[value.len() as i32]);
                    }
                    ast::Literal::Hash(value) => {
                        // Compile each key and value in order
                        for (key, val) in value {
                            self.compile_expr(key)?;
                            self.compile_expr(val)?;
                        }
                        // Emit OPHASH with the number of elements (key-value pairs * 2)
                        self.emit(OPHASH, &[(value.len() * 2) as i32]);
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

                if self.last_instruction_is(OPPOP) {
                    self.remove_last_pop();
                }

                let jump_pos = self.emit(OPJUMP, &[9999]);
                
                let after_consequence_pos = self.current_instructions().len();
                
                self.change_operand(else_jump, after_consequence_pos as i32);
                
                if let Some(alternative) = alternative {
                    for stmt in alternative {
                        self.compile_stmt(stmt)?;
                    }
                    if self.last_instruction_is(OPPOP) {
                        self.remove_last_pop();
                    }
                } else {
                    self.emit(OPNULL, &[]);
                }

                let after_alternative_pos = self.current_instructions().len();
                self.change_operand(jump_pos, after_alternative_pos as i32);
            }
            ast::Expr::Ident(ident) => {
                let symbol = {
                    let symbol_table = self.symbol_table.borrow();
                    symbol_table.resolve(&ident.0)
                };

                if let Some(symbol) = symbol {
                    match symbol.scope {
                        crate::compiler::symbol_table::SymbolScope::Global => {
                            self.emit(OPGETGLOBAL, &[symbol.index as i32]);
                        }
                        crate::compiler::symbol_table::SymbolScope::Local => {
                            self.emit(OPGETLOCAL, &[symbol.index as i32]);
                        }
                    }
                } else {
                    return Err(format!("undefined variable: {}", ident.0));
                }
            }
            ast::Expr::Index(left, index) => {
                self.compile_expr(left)?;
                self.compile_expr(index)?;
                self.emit(OPINDEX, &[]);
            }
            ast::Expr::Function{ params, body } => {
                // Enter a new scope for the function
                self.enter_scope();

                // Define parameters in the new symbol table
                for param in params {
                    self.symbol_table.borrow_mut().define(&param.0);
                }

                self.compile_block_statement(body)?;

                // Replace last OPPOP with OPRETURNVALUE if needed
                if self.last_instruction_is(OPPOP) {
                    self.replace_last_pop_with_return();
                }
                // If the last instruction is not OPRETURNVALUE, emit OPRETURN
                if !self.last_instruction_is(OPRETURNVALUE) {
                    self.emit(OPRETURN, &[]);
                }

                // Gather function metadata
                let num_locals = self.symbol_table.borrow().num_definitions;
                let instructions = self.leave_scope();

                // Create the compiled function object
                let compiled_fn = Object::CompiledFunction(
                    instructions,
                    num_locals,
                    params.len(),
                );

                // Add the compiled function to constants and emit OPCONSTANT
                let const_index = self.add_constant(compiled_fn) as i32;
                self.emit(OPCONSTANT, &[const_index]);
            }
            ast::Expr::Call{ name: _, function: fun, arguments: args } => {
                // Compile the function to call
                self.compile_expr(&fun)?;
                // Compile each argument to the function
                for arg in args {
                    self.compile_expr(arg)?;
                }
                // Emit OPCALL with number of arguments
                self.emit(OPCALL, &[args.len() as i32]);
            }
            _ => {
                return Err(format!("unsupported expression: {:?}", expr));
            }
        }
        Ok(())
    }

    pub fn bytecode(&self) -> Bytecode {
        Bytecode {
            instructions: self.current_instructions().clone(),
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

    pub fn add_instruction(&mut self, ins: &Instructions) -> usize {
        let pos_new_instruction = self.current_instructions().len();
        let mut updated_instructions = self.current_instructions().clone();
        updated_instructions.0.extend_from_slice(&ins.0);

        self.scopes[self.scope_index].instructions = updated_instructions;

        pos_new_instruction
    }

    pub fn current_instructions(&self) -> &Instructions {
        &self.scopes[self.scope_index].instructions
    }

    pub fn set_last_instruction(&mut self, op: Opcode, pos: usize) {
        let previous = self.scopes[self.scope_index].last_instruction.clone();
        let last = EmittedInstruction { opcode: op, position: pos };

        self.scopes[self.scope_index].previous_instruction = previous;
        self.scopes[self.scope_index].last_instruction = last;
    }

    pub fn last_instruction_is(&self, op: Opcode) -> bool {
        if self.current_instructions().len() == 0 {
            return false;
        }
        self.scopes[self.scope_index].last_instruction.opcode == op
    }

    pub fn replace_instruction(&mut self, pos: usize, new_instruction: &Instructions) {
        let ins = &mut self.scopes[self.scope_index].instructions.0;
        for i in 0..new_instruction.0.len() {
            ins[pos + i] = new_instruction.0[i];
        }
    }

    pub fn change_operand(&mut self, op_pos: usize, operand: i32) {
        let instructions = self.current_instructions();
        if op_pos >= instructions.0.len() {
            return;
        }
        let op = instructions.0[op_pos];
        let new_instruction = code::make(op, &[operand]);
        self.replace_instruction(op_pos, &new_instruction);
    }

    pub fn remove_last_pop(&mut self) {
        let last = self.scopes[self.scope_index].last_instruction.clone();
        let previous = self.scopes[self.scope_index].previous_instruction.clone();

        let old = &self.current_instructions().0;
        let new = old[..last.position].to_vec();

        self.scopes[self.scope_index].instructions.0 = new;
        self.scopes[self.scope_index].last_instruction = previous;
    }
    
    pub fn compile_block_statement(&mut self, block_statement: &ast::BlockStmt) -> Result<(), String> {
        for statement in block_statement {
            self.compile_stmt(statement)?;
        }
        Ok(())
    }

    pub fn enter_scope(&mut self) {
        let scope = CompilationScope {
            instructions: Instructions::new(),
            last_instruction: EmittedInstruction::new(),
            previous_instruction: EmittedInstruction::new(),
        };
        self.scopes.push(scope);
        self.scope_index += 1;

        let new_symbol_table = SymbolTable::new_enclosed(Rc::clone(&self.symbol_table));
        self.symbol_table = Rc::new(RefCell::new(new_symbol_table));
    }

    pub fn leave_scope(&mut self) -> Instructions {
        let instructions = self.current_instructions().clone();

        self.scopes.pop();
        if self.scope_index > 0 {
            self.scope_index -= 1;
        }

        // Restore the symbol table to its outer
        let outer = self.symbol_table.borrow().outer.clone();
        if let Some(outer_rc) = outer {
            self.symbol_table = outer_rc;
        }

        instructions
    }

    pub fn replace_last_pop_with_return(&mut self) {
        let last_pos = self.scopes[self.scope_index].last_instruction.position;
        let new_instruction = code::make(OPRETURNVALUE, &[]);
        self.replace_instruction(last_pos, &new_instruction);
    
        self.scopes[self.scope_index].last_instruction.opcode = OPRETURNVALUE;
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
    use std::rc::Rc;
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
        OPSETGLOBAL,
        OPARRAY,
        OPHASH,
        OPINDEX,
        OPCALL,
        OPRETURNVALUE,
        OPRETURN,
        OPGETLOCAL,
        OPSETLOCAL,
    };

    struct CompilerTestCase<'a> {
        input: &'a str,
        expected_constants: Vec<Object>,
        expected_instructions: Vec<Instructions>,
    }

    #[test]
    fn test_integer_arithmetic() {
        let tests = vec![
            CompilerTestCase {
                input: "1 + 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPADD, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1; 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPPOP, &[]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 - 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPSUB, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 * 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPMULT, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 / 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPDIV, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "-1",
                expected_constants: vec![Object::Int(1)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPMINUS, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "+1",
                expected_constants: vec![Object::Int(1)],
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
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPGREATERTHAN, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 >= 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPGREATERTHANEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 < 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPLESSTHAN, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 <= 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPLESSTHANEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 == 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPEQUAL, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "1 != 2",
                expected_constants: vec![Object::Int(1), Object::Int(2)],
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
                expected_constants: vec![Object::Int(1), Object::Int(2)],
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
                expected_constants: vec![Object::Int(1)],
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
                expected_constants: vec![Object::Int(1)],
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

    #[test]
    fn test_functions() {
        let tests = vec![
            CompilerTestCase {
                input: r#"fn() { return 5 + 10 }"#,
                expected_constants: vec![
                    Object::Int(5),
                    Object::Int(10),
                     Object::CompiledFunction(
                         code::Instructions(
                             vec![
                                 code::make(OPCONSTANT, &[0]),
                                 code::make(OPCONSTANT, &[1]),
                                 code::make(OPADD, &[]),
                                 code::make(OPRETURNVALUE, &[]),
                             ]
                             .into_iter()
                             .flat_map(|ins| ins.0)
                             .collect()
                         ),
                         0, // num_locals
                         0, // num_parameters
                     ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: r#"fn() { 5 + 10 }"#,
                expected_constants: vec![
                    Object::Int(5),
                    Object::Int(10),
                     Object::CompiledFunction(
                         code::Instructions(
                             vec![
                                 code::make(OPCONSTANT, &[0]),
                                 code::make(OPCONSTANT, &[1]),
                                 code::make(OPADD, &[]),
                                 code::make(OPRETURNVALUE, &[]),
                             ]
                             .into_iter()
                             .flat_map(|ins| ins.0)
                             .collect()
                         ),
                         0, // num_locals
                         0, // num_parameters
                     ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: r#"fn() { 1; 2 }"#,
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                     Object::CompiledFunction(
                         code::Instructions(
                             vec![
                                 code::make(OPCONSTANT, &[0]),
                                 code::make(OPPOP, &[]),
                                 code::make(OPCONSTANT, &[1]),
                                 code::make(OPRETURNVALUE, &[]),
                             ]
                             .into_iter()
                             .flat_map(|ins| ins.0)
                             .collect()
                         ),
                         0, // num_locals
                         0, // num_parameters
                     ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_function_calls() {
        let tests = vec![
            CompilerTestCase {
                input: "fn() { 24 }();",
                expected_constants: vec![
                    Object::Int(24),
                    Object::CompiledFunction(
                        code::Instructions(
                            [
                                code::make(OPCONSTANT, &[0]),
                                code::make(OPRETURNVALUE, &[]),
                            ]
                            .iter()
                            .flat_map(|ins| ins.0.clone())
                            .collect()
                        ),
                        0, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPCALL, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "
                let noArg = fn() { 24 };
                noArg();
                ",
                expected_constants: vec![
                    Object::Int(24),
                    Object::CompiledFunction(
                        code::Instructions(
                            [
                                code::make(OPCONSTANT, &[0]),
                                code::make(OPRETURNVALUE, &[]),
                            ]
                            .iter()
                            .flat_map(|ins| ins.0.clone())
                            .collect()
                        ),
                        0, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPSETGLOBAL, &[0]),
                    code::make(OPGETGLOBAL, &[0]),
                    code::make(OPCALL, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_functions_without_return_value() {
        let tests = vec![
            CompilerTestCase {
                input: "fn() { }",
                expected_constants: vec![
                    Object::CompiledFunction(
                        code::Instructions(
                            [
                                code::make(OPRETURN, &[]),
                            ]
                            .iter()
                            .flat_map(|ins| ins.0.clone())
                            .collect()
                        ),
                        0, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPPOP, &[]),
                ],
            }
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_conditionals() {
        use crate::code::{self, Instructions, OPCONSTANT, OPTRUE, OPJUMPELSE, OPJUMP, OPNULL, OPPOP};

        struct CompilerTestCase<'a> {
            input: &'a str,
            expected_constants: Vec<Object>,
            expected_instructions: Vec<Instructions>,
        }

        let tests = [
            CompilerTestCase {
                input: "
                if (true) { 10 }; 3333;
                ",
                expected_constants: vec![Object::Int(10), Object::Int(3333)],
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
                expected_constants: vec![Object::Int(10), Object::Int(20), Object::Int(3333)],
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

    #[test]
    fn test_string_expressions() {
        let tests = vec![
            CompilerTestCase {
                input: r#""monkey""#,
                expected_constants: vec![Object::String("monkey".to_string())],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: r#""mon" + "key""#,
                expected_constants: vec![Object::String("mon".to_string()), Object::String("key".to_string())],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPADD, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_array_literals() {
        let tests = vec![
            CompilerTestCase {
                input: "[]",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPARRAY, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "[1, 2, 3]",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPARRAY, &[3]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "[1 + 2, 3 - 4, 5 * 6]",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPADD, &[]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPCONSTANT, &[3]),
                    code::make(OPSUB, &[]),
                    code::make(OPCONSTANT, &[4]),
                    code::make(OPCONSTANT, &[5]),
                    code::make(OPMULT, &[]),
                    code::make(OPARRAY, &[3]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_hash_literals() {
        let tests = vec![
            CompilerTestCase {
                input: "{}",
                expected_constants: vec![],
                expected_instructions: vec![
                    code::make(OPHASH, &[0]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "{1: 2, 3: 4, 5: 6}",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPCONSTANT, &[3]),
                    code::make(OPCONSTANT, &[4]),
                    code::make(OPCONSTANT, &[5]),
                    code::make(OPHASH, &[6]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "{1: 2 + 3, 4: 5 * 6}",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                    Object::Int(5),
                    Object::Int(6),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPADD, &[]),
                    code::make(OPCONSTANT, &[3]),
                    code::make(OPCONSTANT, &[4]),
                    code::make(OPCONSTANT, &[5]),
                    code::make(OPMULT, &[]),
                    code::make(OPHASH, &[4]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_index_expressions() {
        let tests = vec![
            CompilerTestCase {
                input: "[1, 2, 3][1 + 1]",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(1),
                    Object::Int(1),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPARRAY, &[3]),
                    code::make(OPCONSTANT, &[3]),
                    code::make(OPCONSTANT, &[4]),
                    code::make(OPADD, &[]),
                    code::make(OPINDEX, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "{1: 2}[2 - 1]",
                expected_constants: vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(2),
                    Object::Int(1),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPHASH, &[2]),
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPCONSTANT, &[3]),
                    code::make(OPSUB, &[]),
                    code::make(OPINDEX, &[]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }

    #[test]
    fn test_compiler_scopes() {
        let mut compiler = super::Compiler::new();

        // Initially, scope_index should be 0
        assert_eq!(compiler.scope_index, 0, "scope_index wrong. got={}, want=0", compiler.scope_index);

        let global_symbol_table = Rc::clone(&compiler.symbol_table);

        // Emit OPMULT in global scope
        compiler.emit(OPMULT, &[]);

        // Enter new scope
        compiler.enter_scope();
        assert_eq!(compiler.scope_index, 1, "scope_index wrong. got={}, want=1", compiler.scope_index);

        // Emit OPSUB in new scope
        compiler.emit(OPSUB, &[]);

        // There should be 1 instruction in the current scope
        let current_scope = &compiler.scopes[compiler.scope_index];
        assert_eq!(
            current_scope.instructions.len(),
            1,
            "instructions length wrong. got={}",
            current_scope.instructions.len()
        );

        // Last instruction should be OPSUB
        let last = &current_scope.last_instruction;
        assert_eq!(
            last.opcode, OPSUB,
            "last_instruction.opcode wrong. got={:?}, want={:?}",
            last.opcode, OPSUB
        );

        let current_symbol_table_outer = compiler.symbol_table.borrow().outer.clone();
        let global_symbol_table_ptr = Rc::as_ptr(&global_symbol_table);
        let outer_ptr = current_symbol_table_outer
            .as_ref()
            .map(|outer| Rc::as_ptr(outer))
            .unwrap_or(std::ptr::null());
        assert_eq!(
            outer_ptr, global_symbol_table_ptr,
            "compiler did not enclose symbol_table"
        );

        // Leave scope
        compiler.leave_scope();
        assert_eq!(compiler.scope_index, 0, "scope_index wrong. got={}, want=0", compiler.scope_index);

        // Symbol table should be restored to global
        let current_symbol_table_ptr = Rc::as_ptr(&compiler.symbol_table);
        assert_eq!(
            current_symbol_table_ptr, global_symbol_table_ptr,
            "compiler did not restore global symbol table"
        );
        // Global symbol table should not have an outer
        assert!(
            compiler.symbol_table.borrow().outer.is_none(),
            "compiler modified global symbol table incorrectly"
        );

        // Emit OPADD in global scope
        compiler.emit(OPADD, &[]);

        let global_scope = &compiler.scopes[compiler.scope_index];
        assert_eq!(
            global_scope.instructions.len(),
            2,
            "instructions length wrong. got={}",
            global_scope.instructions.len()
        );

        let last = &global_scope.last_instruction;
        assert_eq!(
            last.opcode, OPADD,
            "last_instruction.opcode wrong. got={:?}, want={:?}",
            last.opcode, OPADD
        );

        let previous = &global_scope.previous_instruction;
        assert_eq!(
            previous.opcode, OPMULT,
            "previous_instruction.opcode wrong. got={:?}, want={:?}",
            previous.opcode, OPMULT
        );
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
        expected: &[Object],
        actual: &[Object],
    ) -> Result<(), String> {
        if expected.len() != actual.len() {
            return Err(format!(
                "wrong number of constants. got={}, want={}",
                actual.len(),
                expected.len()
            ));
        }

        for (i, expected_constant) in expected.iter().enumerate() {
            match expected_constant {
                Object::String(expected_str) => {
                    match &actual[i] {
                        Object::String(actual_str) => {
                            if expected_str != actual_str {
                                return Err(format!(
                                    "constant {} - testStringObject failed: expected '{}', got '{}'",
                                    i, expected_str, actual_str
                                ));
                            }
                        }
                        other => {
                            return Err(format!(
                                "constant {} - testStringObject failed: expected String, got {:?}",
                                i, other
                            ));
                        }
                    }
                }
                Object::Int(expected_int) => {
                    match &actual[i] {
                        Object::Int(actual_int) => {
                            if expected_int != actual_int {
                                return Err(format!(
                                    "constant {} - testIntegerObject failed: expected {}, got {}",
                                    i, expected_int, actual_int
                                ));
                            }
                        }
                        other => {
                            return Err(format!(
                                "constant {} - testIntegerObject failed: expected Int, got {:?}",
                                i, other
                            ));
                        }
                    }
                }
                Object::CompiledFunction(expected_fn, 0, 0) => {
                    match &actual[i] {
                        Object::CompiledFunction(actual_fn, 0, 0) => {
                            if let Err(e) = test_instructions(&[expected_fn.clone()], &actual_fn) {
                                return Err(format!(
                                    "constant {} - testInstructions failed: {}",
                                    i, e
                                ));
                            }
                        }
                        other => {
                            return Err(format!(
                                "constant {} - not a function: got {:?}",
                                i, other
                            ));
                        }
                    }
                }
                _ => {
                    // Optionally handle other types or ignore
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_let_statement_scopes() {
        let tests = vec![
            CompilerTestCase {
                input: "
			let num = 55;
			fn() { num }
			",
                expected_constants: vec![
                    Object::Int(55),
                    Object::CompiledFunction(
                        code::Instructions(
                            vec![
                                code::make(OPGETGLOBAL, &[0]),
                                code::make(OPRETURNVALUE, &[]),
                            ]
                            .into_iter()
                            .flat_map(|ins| ins.0)
                            .collect()
                        ),
                        0, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[0]),
                    code::make(OPSETGLOBAL, &[0]),
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "
			fn() {
				let num = 55;
				num
			}
			",
                expected_constants: vec![
                    Object::Int(55),
                    Object::CompiledFunction(
                        code::Instructions(
                            vec![
                                code::make(OPCONSTANT, &[0]),
                                code::make(OPSETLOCAL, &[0]),
                                code::make(OPGETLOCAL, &[0]),
                                code::make(OPRETURNVALUE, &[]),
                            ]
                            .into_iter()
                            .flat_map(|ins| ins.0)
                            .collect()
                        ),
                        1, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[1]),
                    code::make(OPPOP, &[]),
                ],
            },
            CompilerTestCase {
                input: "
			fn() {
				let a = 55;
				let b = 77;
				a + b
			}
			",
                expected_constants: vec![
                    Object::Int(55),
                    Object::Int(77),
                    Object::CompiledFunction(
                        code::Instructions(
                            vec![
                                code::make(OPCONSTANT, &[0]),
                                code::make(OPSETLOCAL, &[0]),
                                code::make(OPCONSTANT, &[1]),
                                code::make(OPSETLOCAL, &[1]),
                                code::make(OPGETLOCAL, &[0]),
                                code::make(OPGETLOCAL, &[1]),
                                code::make(OPADD, &[]),
                                code::make(OPRETURNVALUE, &[]),
                            ]
                            .into_iter()
                            .flat_map(|ins| ins.0)
                            .collect()
                        ),
                        2, // num_locals
                        0, // num_parameters
                    ),
                ],
                expected_instructions: vec![
                    code::make(OPCONSTANT, &[2]),
                    code::make(OPPOP, &[]),
                ],
            },
        ];

        run_compiler_tests(tests);
    }
}
