extern crate monkey;

use monkey::evaluator::builtins::new_builtins;
use monkey::evaluator::env::Env;
use monkey::evaluator::object::Object;
use monkey::evaluator::Evaluator;
use monkey::lexer::Lexer;
use monkey::parser::Parser;
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
                evaluator.define_macros(&mut program);
                let expanded = evaluator.expand_macros(program);
                let errors = parser.errors();

                if errors.len() > 0 {
                    for err in errors {
                        println!("{}", err);
                    }
                    continue;
                }

                if let Some(evaluated) = evaluator.eval(expanded) {
                    println!("{}\n", evaluated);
                }
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
