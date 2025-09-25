use std::fmt;

#[derive(PartialEq, Clone, Debug)]
pub enum Token {
    Illegal,
    EOF,

    // identifier and literals
    Ident(String),
    Int(i64),
    String(String),

    // operators
    Assign,
    Plus,
    Minus,
    Bang,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    GreaterThanEqual,
    LessThanEqual,
    GreaterThan,
    LessThan,

    // punctuations
    Comma,
    SemiColon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Colon,

    // reserved words
    Function,
    Let,
    True,
    False,
    If,
    Else,
    Return,

    // macro
    Macro,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
