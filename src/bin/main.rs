extern crate monkey;

use monkey::compiler::Compiler;
use monkey::evaluator::builtins::new_builtins;
use monkey::evaluator::env::Env;
use monkey::evaluator::object::Object;
use monkey::evaluator::Evaluator;
use monkey::lexer::Lexer;
use monkey::parser::Parser;
use monkey::vm::VM;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("monkey REPL 0.1.0");

    let mut rl = Editor::<()>::new();
    let mut env = Env::from(new_builtins());

    env.set(
        String::from("puts"),
        &Object::Builtin(-1, |args| {
            for arg in args {
                println!("{}", arg);
            }
            Object::Null
        }),
    );

    let mut evaluator = Evaluator::new(Rc::new(RefCell::new(env)));

    loop {
        match rl.readline(">> ") {
            Ok(line) => {
                rl.add_history_entry(&line);

                let mut parser = Parser::new(Lexer::new(&line));
                let mut program = parser.parse();
                let errors = parser.errors();

                if errors.len() > 0 {
                    for err in errors {
                        println!("{}", err);
                    }
                    continue;
                }

                evaluator.define_macros(&mut program);
                let expanded = evaluator.expand_macros(program);
                
                let mut compiler = Compiler::new();
                if let Err(e) = compiler.compile(&expanded) {
                    println!("Compiler error: {}", e);
                    continue;
                }
                
                let bytecode = compiler.bytecode();
                let mut vm = VM::new(&bytecode);
                
                if let Err(e) = vm.run() {
                    println!("VM error: {}", e);
                    continue;
                }

                let stack_top = vm.last_popped_stack_elem();
                println!("{}\n", stack_top);
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nExiting...");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
            }
        }
    }
}
