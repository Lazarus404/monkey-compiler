pub mod builtins;
pub mod env;
pub mod object;

use crate::parser::ast::*;
use env::*;
use object::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct Evaluator {
    env: Rc<RefCell<Env>>,
}

impl Evaluator {
    pub fn new(env: Rc<RefCell<Env>>) -> Self {
        Evaluator { env }
    }

    pub fn eval(&mut self, program: Program) -> Option<Object> {
        let mut result = None;

        for stmt in program {
            match self.eval_stmt(stmt) {
                Some(Object::ReturnValue(value)) => return Some(*value),
                Some(Object::Error(msg)) => return Some(Object::Error(msg)),
                obj => result = obj,
            }
        }

        result
    }

    fn error(msg: String) -> Object {
        Object::Error(msg)
    }

    fn is_error(obj: &Object) -> bool {
        match obj {
            Object::Error(_) => true,
            _ => false,
        }
    }

    fn is_truthy(obj: Object) -> bool {
        match obj {
            Object::Null | Object::Bool(false) => false,
            _ => true,
        }
    }

    fn eval_stmt(&mut self, stmt: Stmt) -> Option<Object> {
        match stmt {
            Stmt::Let(ident, expr) => {
                let value = match self.eval_expr(expr) {
                    Some(value) => value,
                    None => return None,
                };
                if Self::is_error(&value) {
                    Some(value)
                } else {
                    let Identity(name) = ident;
                    self.env.borrow_mut().set(name, &value);
                    None
                }
            }
            Stmt::Expr(expr) => self.eval_expr(expr),
            Stmt::Return(expr) => {
                let value = match self.eval_expr(expr) {
                    Some(value) => value,
                    None => return None,
                };
                if Self::is_error(&value) {
                    Some(value)
                } else {
                    Some(Object::ReturnValue(Box::new(value)))
                }
            }
        }
    }

    fn eval_block_stmt(&mut self, stmts: BlockStmt) -> Option<Object> {
        let mut result = None;

        for stmt in stmts {
            match self.eval_stmt(stmt) {
                Some(Object::ReturnValue(value)) => return Some(Object::ReturnValue(value)),
                Some(Object::Error(msg)) => return Some(Object::Error(msg)),
                obj => result = obj,
            }
        }

        result
    }

    fn eval_expr(&mut self, expr: Expr) -> Option<Object> {
        match expr {
            Expr::Ident(ident) => Some(self.eval_ident(ident)),
            Expr::Lit(literal) => Some(self.eval_literal(literal)),
            Expr::Prefix(prefix, right_expr) => match self.eval_expr(*right_expr) {
                Some(right) => Some(self.eval_prefix_expr(prefix, right)),
                None => None,
            },
            Expr::Infix(infix, left_expr, right_expr) => {
                let left = self.eval_expr(*left_expr);
                let right = self.eval_expr(*right_expr);
                if left.is_some() && right.is_some() {
                    Some(self.eval_infix_expr(infix, left.unwrap(), right.unwrap()))
                } else {
                    None
                }
            }
            Expr::If {
                cond,
                consequence,
                alternative,
            } => self.eval_if_expr(*cond, consequence, alternative),
            Expr::Index(left_expr, index_expr) => {
                let left = self.eval_expr(*left_expr);
                let index = self.eval_expr(*index_expr);
                if left.is_some() && index.is_some() {
                    Some(self.eval_index_expr(left.unwrap(), index.unwrap()))
                } else {
                    None
                }
            }
            Expr::Function { params, body } => {
                Some(Object::Function(params, body, Rc::clone(&self.env)))
            }
            Expr::Call {
                name,
                function,
                arguments,
            } => Some(self.eval_call_expr(name, function, arguments)),
            _ => None,
        }
    }

    fn eval_ident(&mut self, ident: Identity) -> Object {
        let Identity(name) = ident;

        match self.env.borrow_mut().get(name.clone()) {
            Some(value) => value,
            None => Object::Error(String::from(format!("identifier not found: {}", name))),
        }
    }

    fn eval_literal(&mut self, literal: Literal) -> Object {
        match literal {
            Literal::Int(value) => Object::Int(value),
            Literal::Bool(value) => Object::Bool(value),
            Literal::String(value) => Object::String(value),
            Literal::Array(objects) => self.eval_array_literal(objects),
            Literal::Hash(pairs) => self.eval_hash_literal(pairs),
            Literal::Macro(lit) => Object::Macro(lit),
        }
    }

    fn eval_prefix_expr(&mut self, prefix: Prefix, right: Object) -> Object {
        match prefix {
            Prefix::Not => self.eval_not_op_expr(right),
            Prefix::PreMinus => self.eval_minus_prefix_op_expr(right),
            Prefix::PrePlus => self.eval_plus_prefix_op_expr(right),
        }
    }

    fn eval_not_op_expr(&mut self, right: Object) -> Object {
        match right {
            Object::Bool(true) => Object::Bool(false),
            Object::Bool(false) => Object::Bool(true),
            Object::Null => Object::Bool(true),
            _ => Object::Bool(false),
        }
    }

    fn eval_minus_prefix_op_expr(&mut self, right: Object) -> Object {
        match right {
            Object::Int(value) => Object::Int(-value),
            _ => Self::error(format!("unknown operator: -{}", right)),
        }
    }

    fn eval_plus_prefix_op_expr(&mut self, right: Object) -> Object {
        match right {
            Object::Int(value) => Object::Int(value),
            _ => Self::error(format!("unknown operator: {}", right)),
        }
    }

    fn eval_infix_expr(&mut self, infix: Infix, left: Object, right: Object) -> Object {
        match left {
            Object::Int(left_value) => {
                if let Object::Int(right_value) = right {
                    self.eval_infix_integer_expr(infix, left_value, right_value)
                } else {
                    Self::error(format!("type mismatch: {} {} {}", left, infix, right))
                }
            }
            Object::String(left_value) => {
                if let Object::String(right_value) = right {
                    self.eval_infix_string_expr(infix, left_value, right_value)
                } else {
                    Self::error(format!("type mismatch: {} {} {}", left_value, infix, right))
                }
            }
            Object::Bool(left_value) => {
                if let Object::Bool(right_value) = right {
                    self.eval_infix_boolean_expr(infix, left_value, right_value)
                } else {
                    Self::error(format!("type mismatch: {} {} {}", left_value, infix, right))
                }
            }
            _ => Self::error(format!("unknown operator: {} {} {}", left, infix, right)),
        }
    }

    fn eval_infix_integer_expr(&mut self, infix: Infix, left: i64, right: i64) -> Object {
        match infix {
            Infix::Plus => Object::Int(left + right),
            Infix::Minus => Object::Int(left - right),
            Infix::Multiply => Object::Int(left * right),
            Infix::Divide => Object::Int(left / right),
            Infix::LessThan => Object::Bool(left < right),
            Infix::LessThanEqual => Object::Bool(left <= right),
            Infix::GreaterThan => Object::Bool(left > right),
            Infix::GreaterThanEqual => Object::Bool(left >= right),
            Infix::Equal => Object::Bool(left == right),
            Infix::NotEqual => Object::Bool(left != right),
        }
    }

    fn eval_infix_string_expr(&mut self, infix: Infix, left: String, right: String) -> Object {
        match infix {
            Infix::Plus => Object::String(format!("{}{}", left, right)),
            _ => Object::Error(String::from(format!(
                "unknown operator: {} {} {}",
                left, infix, right
            ))),
        }
    }

    fn eval_infix_boolean_expr(&mut self, infix: Infix, left: bool, right: bool) -> Object {
        match infix {
            Infix::Equal => Object::Bool(left == right),
            Infix::NotEqual => Object::Bool(left != right),
            _ => Object::Error(String::from(format!(
                "unknown operator: {} {} {}",
                left, infix, right
            ))),
        }
    }

    fn eval_if_expr(
        &mut self,
        cond: Expr,
        consequence: BlockStmt,
        alternative: Option<BlockStmt>,
    ) -> Option<Object> {
        let cond = match self.eval_expr(cond) {
            Some(cond) => cond,
            None => return None,
        };

        if Self::is_truthy(cond) {
            self.eval_block_stmt(consequence)
        } else if let Some(alt) = alternative {
            self.eval_block_stmt(alt)
        } else {
            None
        }
    }

    fn eval_call_expr(&mut self, name: Option<String>, func: Box<Expr>, args: Vec<Expr>) -> Object {
        match name {
            Some(name) => {
                if name == String::from("quote") {
                    return self.quote_expr(args[0].clone());
                }
                ()
            }
            _ => (),
        };

        let args = args
            .iter()
            .map(|e| self.eval_expr(e.clone()).unwrap_or(Object::Null))
            .collect::<Vec<_>>();

        let (params, body, env) = match self.eval_expr(*func) {
            Some(Object::Function(params, body, env)) => (params, body, env),
            Some(Object::Builtin(expect_param_num, f)) => {
                if expect_param_num < 0 || expect_param_num == args.len() as i32 {
                    return f(args);
                } else {
                    return Self::error(format!(
                        "wrong number of arguments. got={}, want={}",
                        args.len(),
                        expect_param_num,
                    ));
                }
            }
            Some(o) => return Self::error(format!("{} is not valid function", o)),
            None => return Object::Null,
        };

        if params.len() != args.len() {
            return Self::error(format!(
                "wrong number of arguments: {} expected but {} given",
                params.len(),
                args.len()
            ));
        }

        let current_env = Rc::clone(&self.env);
        let mut scoped_env = Env::new_with_outer(Rc::clone(&env));
        let list = params.iter().zip(args.iter());
        for (_, (ident, o)) in list.enumerate() {
            let Identity(name) = ident.clone();
            scoped_env.set(name, o);
        }

        self.env = Rc::new(RefCell::new(scoped_env));

        let object = self.eval_block_stmt(body);

        self.env = current_env;

        match object {
            Some(o) => o,
            None => Object::Null,
        }
    }

    fn eval_index_expr(&mut self, left: Object, index: Object) -> Object {
        match left {
            Object::Array(ref array) => {
                if let Object::Int(i) = index {
                    self.eval_array_index_expr(array.clone(), i)
                } else {
                    Self::error(format!("index operator not supported: {}", left))
                }
            }
            Object::Hash(ref hash) => match index {
                Object::Int(_) | Object::Bool(_) | Object::String(_) => match hash.get(&index) {
                    Some(o) => o.clone(),
                    None => Object::Null,
                },
                Object::Error(_) => index,
                _ => Self::error(format!("unusable as hash key: {}", index)),
            },
            _ => Self::error(format!("uknown operator: {} {}", left, index)),
        }
    }

    fn eval_array_index_expr(&mut self, array: Vec<Object>, index: i64) -> Object {
        let max = array.len() as i64;

        if index < 0 || index > max {
            return Object::Null;
        }

        match array.get(index as usize) {
            Some(o) => o.clone(),
            None => Object::Null,
        }
    }

    fn eval_array_literal(&mut self, objects: Vec<Expr>) -> Object {
        Object::Array(
            objects
                .iter()
                .map(|e| self.eval_expr(e.clone()).unwrap_or(Object::Null))
                .collect::<Vec<_>>(),
        )
    }

    fn eval_hash_literal(&mut self, pairs: Vec<(Expr, Expr)>) -> Object {
        let mut hash = HashMap::new();

        for (key_expr, value_expr) in pairs {
            let key = self.eval_expr(key_expr).unwrap_or(Object::Null);
            if Self::is_error(&key) {
                return key;
            }

            let value = self.eval_expr(value_expr).unwrap_or(Object::Null);
            if Self::is_error(&value) {
                return value;
            }

            hash.insert(key, value);
        }

        Object::Hash(hash)
    }

    fn quote_expr(&mut self, expr: Expr) -> Object {
        let n = self.eval_unquote(&Node::Expr(expr));
        Object::Quote(n)
    }

    fn eval_unquote(&mut self, quoted: &Node) -> Node {
        modify(
            quoted.clone(),
            Rc::new(RefCell::new(|node: Node| -> Node {
                if !self.is_unquote_call(&node) {
                    return node;
                }

                if let Node::Expr(Expr::Call {
                    name: _,
                    function: _,
                    arguments,
                }) = &node
                {
                    if arguments.len() != 1 {
                        return node;
                    }
                    match self.eval_expr(arguments[0].clone()) {
                        Some(unquoted) => self.convert_object_to_ast_node(unquoted),
                        None => node,
                    }
                } else {
                    node
                }
            })),
        )
    }

    fn convert_object_to_ast_node(&mut self, obj: Object) -> Node {
        match obj {
            Object::Int(int) => Node::Expr(Expr::Lit(Literal::Int(int))),
            Object::Bool(boolean) => Node::Expr(Expr::Lit(Literal::Bool(boolean))),
            Object::Quote(quote) => quote,
            _ => unimplemented!(),
        }
    }

    fn is_unquote_call(&mut self, node: &Node) -> bool {
        if let Node::Expr(Expr::Call {
            name: _,
            function,
            arguments: _,
        }) = node
        {
            if let Expr::Ident(Identity(ident)) = &**function {
                return ident == "unquote";
            };
            false
        } else {
            false
        }
    }

    pub fn define_macros(&mut self, program: &mut Program) {
        let mut def = vec![];
        for (i, stmt) in program.iter().enumerate() {
            if self.is_macro_definition(stmt) {
                self.add_macro(stmt);
                def.push(i);
            }
        }

        def.reverse();
        for def_indx in def.iter() {
            program.remove(*def_indx);
        }
    }

    fn is_macro_definition(&mut self, node: &Stmt) -> bool {
        if let Stmt::Let(_ident, expr) = node {
            if let Expr::Lit(Literal::Macro(_)) = expr {
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    fn add_macro(&mut self, stmt: &Stmt) {
        if let Stmt::Let(Identity(ident), expr) = stmt {
            if let Expr::Lit(Literal::Macro(macro_lit)) = &expr {
                let macro_o = Object::Macro(MacroLiteral {
                    params: macro_lit.params.clone(),
                    body: macro_lit.body.clone(),
                });
                self.env.borrow_mut().set(ident.clone(), &macro_o);
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }

    pub fn expand_macros(&mut self, program: Program) -> Program {
        if let Node::Program(p) = modify(
            Node::Program(program),
            Rc::new(RefCell::new(|node: Node| -> Node {
                if let Node::Expr(Expr::Call {
                    name: _,
                    function: _,
                    arguments: _,
                }) = &node
                {
                    if let Node::Expr(call) = &node {
                        if let Some(macro_o) = self.is_macro_call(call) {
                            let args = self.quote_args(call);
                            for (i, Identity(param)) in macro_o.params.iter().enumerate() {
                                self.env.borrow_mut().set(param.clone(), &args[i]);
                            }
                            let evaluated = self.eval(*macro_o.body);
                            if let Some(Object::Quote(Node::Expr(quote))) = evaluated {
                                Node::Expr(quote)
                            } else {
                                panic!("we only support returning AST-nodes from macros");
                            }
                        } else {
                            node
                        }
                    } else {
                        node
                    }
                } else {
                    node
                }
            })),
        ) {
            p
        } else {
            unreachable!()
        }
    }

    fn is_macro_call(&mut self, exp: &Expr) -> Option<MacroLiteral> {
        if let Expr::Call {
            name: _,
            function,
            arguments: _,
        } = exp
        {
            if let Expr::Ident(Identity(identifier)) = &**function {
                if let Some(obj) = self.env.borrow_mut().get(identifier.clone()) {
                    if let Object::Macro(macro_o) = obj {
                        Some(macro_o)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn quote_args(&mut self, exp: &Expr) -> Vec<Object> {
        let mut args = vec![];
        if let Expr::Call {
            name: _,
            function: _,
            arguments,
        } = exp
        {
            for a in arguments {
                args.push(Object::Quote(Node::Expr(a.clone())));
            }
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::builtins::new_builtins;
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn eval(input: &str) -> Option<Object> {
        Evaluator::new(Rc::new(RefCell::new(Env::from(new_builtins()))))
            .eval(Parser::new(Lexer::new(input)).parse())
    }

    fn assert_eq(tests: Vec<(&str, Option<Object>)>) {
        for (input, expect) in tests {
            assert_eq!(expect, eval(input));
        }
    }

    #[test]
    fn test_int_expr() {
        let tests = vec![
            ("5", Some(Object::Int(5))),
            ("10", Some(Object::Int(10))),
            ("-5", Some(Object::Int(-5))),
            ("-10", Some(Object::Int(-10))),
            ("+5", Some(Object::Int(5))),
            ("+10", Some(Object::Int(10))),
            ("+(-5)", Some(Object::Int(-5))),
            ("+(-10)", Some(Object::Int(-10))),
            ("5 + 5 + 5 + 5 - 10", Some(Object::Int(10))),
            ("2 * 2 * 2 * 2 * 2", Some(Object::Int(32))),
            ("-50 + 100 + -50", Some(Object::Int(0))),
            ("5 * 2 + 10", Some(Object::Int(20))),
            ("5 + 2 * 10", Some(Object::Int(25))),
            ("20 + 2 * -10", Some(Object::Int(0))),
            ("50 / 2 * 2 + 10", Some(Object::Int(60))),
            ("2 * (5 + 10)", Some(Object::Int(30))),
            ("3 * 3 * 3 + 10", Some(Object::Int(37))),
            ("3 * (3 * 3) + 10", Some(Object::Int(37))),
            ("(5 + 10 * 2 + 15 / 3) * 2 + -10", Some(Object::Int(50))),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_bool_expr() {
        let tests = vec![
            ("true", Some(Object::Bool(true))),
            ("false", Some(Object::Bool(false))),
            ("1 < 2", Some(Object::Bool(true))),
            ("1 > 2", Some(Object::Bool(false))),
            ("1 < 1", Some(Object::Bool(false))),
            ("1 > 1", Some(Object::Bool(false))),
            ("1 >= 1", Some(Object::Bool(true))),
            ("1 <= 1", Some(Object::Bool(true))),
            ("1 >= 2", Some(Object::Bool(false))),
            ("1 <= 1", Some(Object::Bool(true))),
            ("2 <= 1", Some(Object::Bool(false))),
            ("1 == 1", Some(Object::Bool(true))),
            ("1 != 1", Some(Object::Bool(false))),
            ("1 == 2", Some(Object::Bool(false))),
            ("1 != 2", Some(Object::Bool(true))),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_not_op() {
        let tests = vec![
            ("!true", Some(Object::Bool(false))),
            ("!false", Some(Object::Bool(true))),
            ("!!true", Some(Object::Bool(true))),
            ("!!false", Some(Object::Bool(false))),
            ("!!5", Some(Object::Bool(true))),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_if_else_expr() {
        let tests = vec![
            ("if (true) { 10 }", Some(Object::Int(10))),
            ("if (false) { 10 }", None),
            ("if (1) { 10 }", Some(Object::Int(10))),
            ("if (1 < 2) { 10 }", Some(Object::Int(10))),
            ("if (1 > 2) { 10 }", None),
            ("if (1 > 2) { 10 } else { 20 }", Some(Object::Int(20))),
            ("if (1 < 2) { 10 } else { 20 }", Some(Object::Int(10))),
            ("if (1 <= 2) { 10 }", Some(Object::Int(10))),
            ("if (1 >= 2) { 10 }", None),
            ("if (1 >= 2) { 10 } else { 20 }", Some(Object::Int(20))),
            ("if (1 <= 2) { 10 } else { 20 }", Some(Object::Int(10))),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_return_stmt() {
        let tests = vec![
            ("return 10;", Some(Object::Int(10))),
            ("return 10; 9;", Some(Object::Int(10))),
            ("return 2 * 5; 9;", Some(Object::Int(10))),
            ("9; return 2 * 5; 9;", Some(Object::Int(10))),
            (
                r#"
            if (10 > 1) {
              if (10 > 1) {
                return 10;
              }
              return 1;
            }
          "#,
                Some(Object::Int(10)),
            ),
            (
                r#"return if (10 > 1) 10; else 2; return 5;"#,
                Some(Object::Int(10)),
            ),
            (
                r#"return if (10 < 1) 10 else 2; return 5;"#,
                Some(Object::Int(2)),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_let_stmt() {
        let tests = vec![
            ("let a = 5; a;", Some(Object::Int(5))),
            ("let a = 5 * 5; a;", Some(Object::Int(25))),
            ("let a = 5; let b = a; b;", Some(Object::Int(5))),
            (
                "let a = 5; let b = a; let c = a + b + 5; c;",
                Some(Object::Int(15)),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_error_handling() {
        let tests = vec![
            (
                "5 + true",
                Some(Object::Error(String::from("type mismatch: 5 + true"))),
            ),
            (
                "5 + true; 5;",
                Some(Object::Error(String::from("type mismatch: 5 + true"))),
            ),
            (
                "-true",
                Some(Object::Error(String::from("unknown operator: -true"))),
            ),
            (
                "5; true + false; 5;",
                Some(Object::Error(String::from(
                    "unknown operator: true + false",
                ))),
            ),
            (
                "if (10 > 1) { true + false; }",
                Some(Object::Error(String::from(
                    "unknown operator: true + false",
                ))),
            ),
            (
                "\"Hello\" - \"World\"",
                Some(Object::Error(String::from(
                    "unknown operator: Hello - World",
                ))),
            ),
            (
                r#"
            if (10 > 1) {
              if (10 > 1) {
                return true + false;
              }
              return 1;
            }
          "#,
                Some(Object::Error(String::from(
                    "unknown operator: true + false",
                ))),
            ),
            (
                "foobar",
                Some(Object::Error(String::from("identifier not found: foobar"))),
            ),
            (
                "{\"name\": \"Monkey\"}[fn(x) { x }]",
                Some(Object::Error(String::from(
                    "unusable as hash key: fn(x) { ... }",
                ))),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_fn_object() {
        let input = "fn(x) { x + 2; };";

        assert_eq!(
            Some(Object::Function(
                vec![Identity(String::from("x"))],
                vec![Stmt::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Lit(Literal::Int(2))),
                ))],
                Rc::new(RefCell::new(Env::from(new_builtins()))),
            )),
            eval(input),
        );
    }

    #[test]
    fn test_fn_application() {
        let tests = vec![
            (
                "let identity = fn(x) { x; }; identity(5);",
                Some(Object::Int(5)),
            ),
            (
                "let identity = fn(x) { return x; }; identity(5);",
                Some(Object::Int(5)),
            ),
            (
                "let double = fn(x) { x * 2; }; double(5);",
                Some(Object::Int(10)),
            ),
            (
                "let add = fn(x, y) { x + y; }; add(5, 5);",
                Some(Object::Int(10)),
            ),
            (
                "let add = fn(x, y) { x + y; }; add(5 + 5, add(5, 5));",
                Some(Object::Int(20)),
            ),
            ("fn(x) { x; }(5)", Some(Object::Int(5))),
            (
                "fn(a) { let f = fn(b) { a + b }; f(a); }(5);",
                Some(Object::Int(10)),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_closures() {
        let tests = vec![(
            r#"
          let newAdder = fn(x) {
            fn(y) { x + y };
          }

          let addTwo = newAdder(2);
          addTwo(2);
        "#,
            Some(Object::Int(4)),
        )];

        assert_eq(tests);
    }

    #[test]
    fn test_array_literal() {
        let input = "[1, 2 * 2, 3 + 3]";

        assert_eq!(
            Some(Object::Array(vec![
                Object::Int(1),
                Object::Int(4),
                Object::Int(6),
            ])),
            eval(input),
        );
    }

    #[test]
    fn test_array_index_expr() {
        let tests = vec![
            ("[1, 2, 3][0]", Some(Object::Int(1))),
            ("[1, 2, 3][1]", Some(Object::Int(2))),
            ("let i = 0; [1][i]", Some(Object::Int(1))),
            ("[1, 2, 3][1 + 1];", Some(Object::Int(3))),
            ("let myArray = [1, 2, 3]; myArray[2];", Some(Object::Int(3))),
            (
                "let myArray = [1, 2, 3]; myArray[0] + myArray[1] + myArray[2];",
                Some(Object::Int(6)),
            ),
            (
                "let myArray = [1, 2, 3]; let i = myArray[0]; myArray[i];",
                Some(Object::Int(2)),
            ),
            ("[1, 2, 3][3]", Some(Object::Null)),
            ("[1, 2, 3][-1]", Some(Object::Null)),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_hash_literal() {
        let input = r#"
        let two = "two";
        {
          "one": 10 - 9,
          two: 1 + 1,
          "thr" + "ee": 6 / 2,
          4: 4,
          true: 5,
          false: 6
        }
      "#;

        let mut hash = HashMap::new();
        hash.insert(Object::String(String::from("one")), Object::Int(1));
        hash.insert(Object::String(String::from("two")), Object::Int(2));
        hash.insert(Object::String(String::from("three")), Object::Int(3));
        hash.insert(Object::Int(4), Object::Int(4));
        hash.insert(Object::Bool(true), Object::Int(5));
        hash.insert(Object::Bool(false), Object::Int(6));

        assert_eq!(Some(Object::Hash(hash)), eval(input),);
    }

    #[test]
    fn test_hash_index_expr() {
        let tests = vec![
            ("{\"foo\": 5}[\"foo\"]", Some(Object::Int(5))),
            ("{\"foo\": 5}[\"bar\"]", Some(Object::Null)),
            ("let key = \"foo\"; {\"foo\": 5}[key]", Some(Object::Int(5))),
            ("{}[\"foo\"]", Some(Object::Null)),
            ("{5: 5}[5]", Some(Object::Int(5))),
            ("{true: 5}[true]", Some(Object::Int(5))),
            ("{false: 5}[false]", Some(Object::Int(5))),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_builtin_functions() {
        let tests = vec![
            // len
            ("len(\"\")", Some(Object::Int(0))),
            ("len(\"four\")", Some(Object::Int(4))),
            ("len(\"hello world\")", Some(Object::Int(11))),
            ("len([1, 2, 3])", Some(Object::Int(3))),
            (
                "len(1)",
                Some(Object::Error(String::from(
                    "argument to `len` not supported, got 1",
                ))),
            ),
            (
                "len(\"one\", \"two\")",
                Some(Object::Error(String::from(
                    "wrong number of arguments. got=2, want=1",
                ))),
            ),
            // first
            ("first([1, 2, 3])", Some(Object::Int(1))),
            ("first([])", Some(Object::Null)),
            (
                "first([], [])",
                Some(Object::Error(String::from(
                    "wrong number of arguments. got=2, want=1",
                ))),
            ),
            (
                "first(\"string\")",
                Some(Object::Error(String::from(
                    "argument to `first` must be array. got string",
                ))),
            ),
            (
                "first(1)",
                Some(Object::Error(String::from(
                    "argument to `first` must be array. got 1",
                ))),
            ),
            // last
            ("last([1, 2, 3])", Some(Object::Int(3))),
            ("last([])", Some(Object::Null)),
            (
                "last([], [])",
                Some(Object::Error(String::from(
                    "wrong number of arguments. got=2, want=1",
                ))),
            ),
            (
                "last(\"string\")",
                Some(Object::Error(String::from(
                    "argument to `last` must be array. got string",
                ))),
            ),
            (
                "last(1)",
                Some(Object::Error(String::from(
                    "argument to `last` must be array. got 1",
                ))),
            ),
            // rest
            (
                "rest([1, 2, 3, 4])",
                Some(Object::Array(vec![
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                ])),
            ),
            (
                "rest([2, 3, 4])",
                Some(Object::Array(vec![Object::Int(3), Object::Int(4)])),
            ),
            ("rest([4])", Some(Object::Array(vec![]))),
            ("rest([])", Some(Object::Null)),
            (
                "rest([], [])",
                Some(Object::Error(String::from(
                    "wrong number of arguments. got=2, want=1",
                ))),
            ),
            (
                "rest(\"string\")",
                Some(Object::Error(String::from(
                    "argument to `rest` must be array. got string",
                ))),
            ),
            (
                "rest(1)",
                Some(Object::Error(String::from(
                    "argument to `rest` must be array. got 1",
                ))),
            ),
            // push
            (
                "push([1, 2, 3], 4)",
                Some(Object::Array(vec![
                    Object::Int(1),
                    Object::Int(2),
                    Object::Int(3),
                    Object::Int(4),
                ])),
            ),
            ("push([], 1)", Some(Object::Array(vec![Object::Int(1)]))),
            (
                "let a = [1]; push(a, 2); a",
                Some(Object::Array(vec![Object::Int(1)])),
            ),
            (
                "push([], [], [])",
                Some(Object::Error(String::from(
                    "wrong number of arguments. got=3, want=2",
                ))),
            ),
            (
                "push(\"string\", 1)",
                Some(Object::Error(String::from(
                    "argument to `push` must be array. got string",
                ))),
            ),
            (
                "push(1, 1)",
                Some(Object::Error(String::from(
                    "argument to `push` must be array. got 1",
                ))),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_quote() {
        let tests = vec![
            (
                "quote(5)",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Int(5))))),
            ),
            (
                "quote(5 + 8)",
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(8))),
                )))),
            ),
            (
                "quote(foobar + barfoo)",
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Ident(Identity(String::from("foobar")))),
                    Box::new(Expr::Ident(Identity(String::from("barfoo")))),
                )))),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_quote_unquote() {
        let tests = vec![
            (
                "quote(unquote(4))",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Int(4))))),
            ),
            (
                "quote(unquote(4 + 4))",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Int(8))))),
            ),
            (
                "quote(8 + unquote(4 + 4))",
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(8))),
                    Box::new(Expr::Lit(Literal::Int(8))),
                )))),
            ),
            (
                "quote(unquote(4 + 4) + 8)",
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(8))),
                    Box::new(Expr::Lit(Literal::Int(8))),
                )))),
            ),
            (
                "let foobar = 8; quote(foobar)",
                Some(Object::Quote(Node::Expr(Expr::Ident(Identity(
                    "foobar".to_string(),
                ))))),
            ),
            (
                "let foobar = 8; quote(unquote(foobar))",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Int(8))))),
            ),
            (
                "quote(unquote(true))",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Bool(true))))),
            ),
            (
                "quote(unquote(true == false))",
                Some(Object::Quote(Node::Expr(Expr::Lit(Literal::Bool(false))))),
            ),
            (
                "quote(unquote(quote(4 + 4)))",
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(4))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )))),
            ),
            (
                r#"let quotedInfixExpression = quote(4 + 4);
              quote(unquote(4 + 4) + unquote(quotedInfixExpression))"#,
                Some(Object::Quote(Node::Expr(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(8))),
                    Box::new(Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Lit(Literal::Int(4))),
                        Box::new(Expr::Lit(Literal::Int(4))),
                    )),
                )))),
            ),
        ];

        assert_eq(tests);
    }

    #[test]
    fn test_define_macros() {
        let input = r#"
        let number = 1;
        let function = fn(x, y) { x + y; };
        let mymacro = macro(x, y) { x + y; };
        "#;

        let env = Env::from(new_builtins());
        let mut eval = Evaluator::new(Rc::new(RefCell::new(env)));
        let mut program = test_parse_program(input);

        eval.define_macros(&mut program);

        assert_eq!(program.len(), 2);

        let mut borrowed_env = eval.env.borrow_mut();
        assert!(borrowed_env.get("number".to_string()).is_none());
        assert!(borrowed_env.get("function".to_string()).is_none());

        if let Some(obj) = borrowed_env.get("mymacro".to_string()) {
            if let Object::Macro(macro_o) = obj {
                assert_eq!(macro_o.params.len(), 2);
                assert_eq!(macro_o.params[0], Identity("x".to_string()));
                assert_eq!(macro_o.params[1], Identity("y".to_string()));
                assert_eq!(
                    macro_o.body,
                    Box::new(vec![Stmt::Expr(Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Ident(Identity(String::from("x")))),
                        Box::new(Expr::Ident(Identity(String::from("y"))))
                    ))])
                );
            } else {
                assert!(false, "object is not Macro")
            }
        } else {
            assert!(false, "macro not in environment.")
        }
    }

    #[test]
    fn test_expand_macros() {
        let tests = [
            (
                r#"
                let infixExpression = macro() { quote(1 + 2); };
                infixExpression();
                "#,
                "(1 + 2)",
            ),
            (
                r#"
                let reverse = macro(a, b) { quote(unquote(b) - unquote(a)); };
                reverse(2 + 2, 10 - 5);
                "#,
                "(10 - 5) - (2 + 2)",
            ),
            (
                r#"
                let unless = macro(condition, consequence, alternative) {
                    quote(if (!(unquote(condition))){
                        unquote(consequence);
                    } else {
                        unquote(alternative);
                    })
                };
                unless(10 > 5, puts("not greater"), puts("greater"));
                "#,
                r#"if (!(10 > 5)) { puts("not greater") } else { puts("greater") }"#,
            ),
        ];

        for (input, expected_str) in tests.iter() {
            let expected = test_parse_program(expected_str);
            let env = Env::from(new_builtins());
            let mut eval = Evaluator::new(Rc::new(RefCell::new(env)));
            let mut program = test_parse_program(input);
            eval.define_macros(&mut program);

            let expanded = eval.expand_macros(program);

            assert_eq!(expanded, expected);
        }
    }

    fn test_parse_program(input: &str) -> Program {
        let l = Lexer::new(input);
        let mut p = Parser::new(l);
        p.parse()
    }
}
