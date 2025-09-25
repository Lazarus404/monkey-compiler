use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
pub type Program = BlockStmt;

#[derive(PartialEq, Eq, Debug, Clone)]
pub struct Identity(pub String);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroLiteral {
    pub params: Vec<Identity>,
    pub body: Box<BlockStmt>,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Node {
    Program(Program),
    BlockStmt(BlockStmt),
    Stmt(Stmt),
    Expr(Expr),
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Stmt {
    Let(Identity, Expr),
    Return(Expr),
    Expr(Expr),
}

pub type BlockStmt = Vec<Stmt>;

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Expr {
    Ident(Identity),
    Lit(Literal),
    Prefix(Prefix, Box<Expr>),
    Infix(Infix, Box<Expr>, Box<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Hash(Vec<(Literal, Expr)>),
    If {
        cond: Box<Expr>,
        consequence: BlockStmt,
        alternative: Option<BlockStmt>,
    },
    Function {
        params: Vec<Identity>,
        body: BlockStmt,
    },
    Call {
        name: Option<String>,
        function: Box<Expr>,
        arguments: Vec<Expr>,
    },
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    String(String),
    Array(Vec<Expr>),
    Hash(Vec<(Expr, Expr)>),
    Macro(MacroLiteral),
}

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PEquals,      // ==
    PLessGreater, // > or <
    PSum,         // +
    PProduct,     // *
    PPrefix,      // !x -x +x
    PCall,        // func(x)
    PIndex,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Prefix {
    PrePlus,
    PreMinus,
    Not,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Infix {
    Plus,
    Minus,
    Divide,
    Multiply,
    Equal,
    NotEqual,
    GreaterThanEqual,
    LessThanEqual,
    GreaterThan,
    LessThan,
}

impl fmt::Display for Infix {
    fn fmt(&self, formatting: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Infix::Plus => write!(formatting, "+"),
            Infix::Minus => write!(formatting, "-"),
            Infix::Divide => write!(formatting, "/"),
            Infix::Multiply => write!(formatting, "*"),
            Infix::Equal => write!(formatting, "=="),
            Infix::NotEqual => write!(formatting, "!="),
            Infix::GreaterThanEqual => write!(formatting, ">="),
            Infix::LessThanEqual => write!(formatting, "<="),
            Infix::GreaterThan => write!(formatting, ">"),
            Infix::LessThan => write!(formatting, "<"),
        }
    }
}

pub fn modify<P: FnMut(Node) -> Node>(target: Node, modifier: Rc<RefCell<P>>) -> Node {
    match target {
        Node::Program(mut program) => {
            for i in 0..program.len() {
                let statement = program.swap_remove(i);
                let modified = modify(Node::Stmt(statement), Rc::clone(&modifier));
                if let Node::Stmt(stmt) = modified {
                    program.insert(i, stmt);
                } else if let Node::Expr(expr) = modified {
                    program.insert(i, Stmt::Expr(expr));
                }
            }
            (&mut *modifier.borrow_mut())(Node::Program(program))
        }
        Node::BlockStmt(mut node) => {
            for i in 0..node.len() {
                let statement = node.swap_remove(i);
                if let Node::Expr(expr) = modify(Node::Stmt(statement), Rc::clone(&modifier)) {
                    node.insert(i, Stmt::Expr(expr));
                }
            }
            (&mut *modifier.borrow_mut())(Node::BlockStmt(node))
        }
        Node::Stmt(Stmt::Expr(node)) => {
            if let Node::Expr(expr) = modify(Node::Expr(node), Rc::clone(&modifier)) {
                (&mut *modifier.borrow_mut())(Node::Expr(expr))
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::Infix(op, left, right)) => {
            if let Node::Expr(left) = modify(Node::Expr(*left), Rc::clone(&modifier)) {
                if let Node::Expr(right) = modify(Node::Expr(*right), Rc::clone(&modifier)) {
                    (&mut *modifier.borrow_mut())(Node::Expr(Expr::Infix(
                        op,
                        Box::new(left),
                        Box::new(right),
                    )))
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::Prefix(op, right)) => {
            if let Node::Expr(right) = modify(Node::Expr(*right), Rc::clone(&modifier)) {
                (&mut *modifier.borrow_mut())(Node::Expr(Expr::Prefix(op, Box::new(right))))
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::Index(left, index)) => {
            if let Node::Expr(left) = modify(Node::Expr(*left), Rc::clone(&modifier)) {
                if let Node::Expr(index) = modify(Node::Expr(*index), Rc::clone(&modifier)) {
                    (&mut *modifier.borrow_mut())(Node::Expr(Expr::Index(
                        Box::new(left),
                        Box::new(index),
                    )))
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::If {
            cond,
            consequence,
            alternative,
        }) => {
            if let Node::Expr(cond) = modify(Node::Expr(*cond), Rc::clone(&modifier)) {
                if let Node::BlockStmt(cons) =
                    modify(Node::BlockStmt(consequence), Rc::clone(&modifier))
                {
                    if alternative.is_some() {
                        if let Node::BlockStmt(alt) =
                            modify(Node::BlockStmt(alternative.unwrap()), Rc::clone(&modifier))
                        {
                            (&mut *modifier.borrow_mut())(Node::Expr(Expr::If {
                                cond: Box::new(cond),
                                consequence: cons,
                                alternative: Some(alt),
                            }))
                        } else {
                            unreachable!()
                        }
                    } else {
                        (&mut *modifier.borrow_mut())(Node::Expr(Expr::If {
                            cond: Box::new(cond),
                            consequence: cons,
                            alternative: None,
                        }))
                    }
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::Function { mut params, body }) => {
            for i in 0..params.len() {
                let identifier = params.swap_remove(i);
                if let Node::Expr(Expr::Ident(ident)) =
                    modify(Node::Expr(Expr::Ident(identifier)), Rc::clone(&modifier))
                {
                    params.insert(i, ident);
                }
            }
            if let Node::BlockStmt(body) = modify(Node::BlockStmt(body), Rc::clone(&modifier)) {
                (&mut *modifier.borrow_mut())(Node::Expr(Expr::Function { params, body }))
            } else {
                unreachable!()
            }
        }
        Node::Stmt(Stmt::Return(value)) => {
            if let Node::Expr(return_value) = modify(Node::Expr(value), Rc::clone(&modifier)) {
                (&mut *modifier.borrow_mut())(Node::Stmt(Stmt::Return(return_value)))
            } else {
                unreachable!()
            }
        }
        Node::Stmt(Stmt::Let(ident, expr)) => {
            if let Node::Expr(value) = modify(Node::Expr(expr), Rc::clone(&modifier)) {
                (&mut *modifier.borrow_mut())(Node::Stmt(Stmt::Let(ident, value)))
            } else {
                unreachable!()
            }
        }
        Node::Expr(Expr::Lit(Literal::Array(mut node))) => {
            for i in 0..node.len() {
                let element = node.swap_remove(i);
                if let Node::Expr(elem) = modify(Node::Expr(element), Rc::clone(&modifier)) {
                    node.insert(i, elem);
                }
            }
            (&mut *modifier.borrow_mut())(Node::Expr(Expr::Lit(Literal::Array(node))))
        }
        Node::Expr(Expr::Lit(Literal::Hash(mut node))) => {
            for i in 0..node.len() {
                let (key, value) = node.swap_remove(i);
                if let Node::Expr(k) = modify(Node::Expr(key), Rc::clone(&modifier)) {
                    if let Node::Expr(v) = modify(Node::Expr(value), Rc::clone(&modifier)) {
                        node.insert(i, (k, v));
                    }
                }
            }
            (&mut *modifier.borrow_mut())(Node::Expr(Expr::Lit(Literal::Hash(node))))
        }
        _ => (&mut *modifier.borrow_mut())(target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one() -> Expr {
        Expr::Lit(Literal::Int(1))
    }
    fn two() -> Expr {
        Expr::Lit(Literal::Int(2))
    }
    fn turn_one_into_two(node: Node) -> Node {
        if let Node::Expr(Expr::Lit(Literal::Int(integer))) = &node {
            if *integer != 1 {
                return node;
            }
            Node::Expr(Expr::Lit(Literal::Int(2)))
        } else {
            node
        }
    }

    #[test]
    fn test_modify() {
        let mut tests = vec![
            (Node::Expr(one()), Node::Expr(two())),
            (
                Node::Program(vec![Stmt::Expr(one())]),
                Node::Program(vec![Stmt::Expr(two())]),
            ),
            (
                Node::Expr(Expr::Infix(Infix::Plus, Box::new(one()), Box::new(two()))),
                Node::Expr(Expr::Infix(Infix::Plus, Box::new(two()), Box::new(two()))),
            ),
            (
                Node::Expr(Expr::Infix(Infix::Plus, Box::new(two()), Box::new(one()))),
                Node::Expr(Expr::Infix(Infix::Plus, Box::new(two()), Box::new(two()))),
            ),
            (
                Node::Expr(Expr::Prefix(Prefix::PreMinus, Box::new(one()))),
                Node::Expr(Expr::Prefix(Prefix::PreMinus, Box::new(two()))),
            ),
            (
                Node::Expr(Expr::Index(Box::new(one()), Box::new(one()))),
                Node::Expr(Expr::Index(Box::new(two()), Box::new(two()))),
            ),
            (
                Node::Expr(Expr::If {
                    cond: Box::new(one()),
                    consequence: vec![Stmt::Expr(one())],
                    alternative: Some(vec![Stmt::Expr(one())]),
                }),
                Node::Expr(Expr::If {
                    cond: Box::new(two()),
                    consequence: vec![Stmt::Expr(two())],
                    alternative: Some(vec![Stmt::Expr(two())]),
                }),
            ),
            (
                Node::Stmt(Stmt::Return(one())),
                Node::Stmt(Stmt::Return(two())),
            ),
            (
                Node::Stmt(Stmt::Let(Identity(String::from("value")), one())),
                Node::Stmt(Stmt::Let(Identity(String::from("value")), two())),
            ),
            (
                Node::Expr(Expr::Function {
                    params: vec![],
                    body: vec![Stmt::Expr(one())],
                }),
                Node::Expr(Expr::Function {
                    params: vec![],
                    body: vec![Stmt::Expr(two())],
                }),
            ),
            (
                Node::Expr(Expr::Lit(Literal::Array(vec![one(), one()]))),
                Node::Expr(Expr::Lit(Literal::Array(vec![two(), two()]))),
            ),
            (
                Node::Expr(Expr::Lit(Literal::Hash(vec![(one(), one())]))),
                Node::Expr(Expr::Lit(Literal::Hash(vec![(two(), two())]))),
            ),
        ];

        while !tests.is_empty() {
            let (input, expected) = tests.pop().unwrap();
            let modified = modify(input, Rc::new(RefCell::new(turn_one_into_two)));
            assert_eq!(modified, expected);
        }
    }
}
