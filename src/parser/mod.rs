pub mod ast;

use crate::lexer::Lexer;
use crate::token::Token;
use ast::*;
use std::fmt;
use std::mem::discriminant as variant;

#[derive(Clone, Debug)]
pub enum ParserErrorType {
    UnexpectedToken,
}

impl fmt::Display for ParserErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ParserErrorType::UnexpectedToken => write!(f, "Unexpected Token"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParserError {
    kind: ParserErrorType,
    description: String,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.description)
    }
}

impl ParserError {
    fn new(kind: ParserErrorType, description: String) -> Self {
        ParserError { kind, description }
    }
}

pub type ParserErrors = Vec<ParserError>;

pub struct Parser<'a> {
    lexer: Lexer<'a>,
    current_token: Token,
    next_token: Token,
    errors: ParserErrors,
}

impl<'a> Parser<'a> {
    pub fn new(lexer: Lexer<'a>) -> Self {
        let mut parser = Parser {
            lexer,
            current_token: Token::EOF,
            next_token: Token::EOF,
            errors: vec![],
        };

        parser.consume();
        parser.consume();

        parser
    }

    pub fn errors(&mut self) -> ParserErrors {
        self.errors.clone()
    }

    pub fn parse(&mut self) -> Program {
        let mut program: Program = vec![];

        while self.current_token != Token::EOF {
            match self.parse_stmt() {
                Some(stmt) => program.push(stmt),
                None => (),
            }
            self.consume();
        }

        program
    }

    fn consume(&mut self) {
        self.current_token = self.next_token.clone();
        self.next_token = self.lexer.next_token();
    }

    fn current_token_is(&mut self, t: Token) -> bool {
        variant(&self.current_token) == variant(&t)
    }

    fn peek_token_is(&mut self, t: &Token) -> bool {
        variant(&self.next_token) == variant(t)
    }

    fn expect_peek(&mut self, t: Token) -> bool {
        match self.peek_token_is(&t) {
            true => {
                self.consume();
                return true;
            }
            _ => {
                self.peek_error(t);
                return false;
            }
        }
    }

    fn peek_error(&mut self, t: Token) {
        let desc = format!(
            "expected next token to be {:?}, got {:?} instead",
            t, self.next_token
        );
        self.errors
            .push(ParserError::new(ParserErrorType::UnexpectedToken, desc));
    }

    fn token_precedences(tok: &Token) -> Precedence {
        match tok {
            Token::Equal | Token::NotEqual => Precedence::PEquals,
            Token::LessThan | Token::LessThanEqual => Precedence::PLessGreater,
            Token::GreaterThan | Token::GreaterThanEqual => Precedence::PLessGreater,
            Token::Plus | Token::Minus => Precedence::PSum,
            Token::Divide | Token::Multiply => Precedence::PProduct,
            Token::LBracket => Precedence::PIndex,
            Token::LParen => Precedence::PCall,
            _ => Precedence::PLowest,
        }
    }

    fn current_precedence(&mut self) -> Precedence {
        Self::token_precedences(&self.current_token)
    }

    fn peek_precedence(&mut self) -> Precedence {
        Self::token_precedences(&self.next_token)
    }

    fn error_no_prefix_parse(&mut self) {
        self.errors.push(ParserError::new(
            ParserErrorType::UnexpectedToken,
            format!(
                "no prefix parse function for {:?} found",
                self.current_token,
            ),
        ));
    }

    fn parse_stmt(&mut self) -> Option<Stmt> {
        match self.current_token {
            Token::Let => self.parse_let_stmt(),
            Token::Return => self.parse_return_stmt(),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_ident(&mut self) -> Option<Identity> {
        match self.current_token {
            Token::Ident(ref mut ident) => Some(Identity(ident.clone())),
            _ => None,
        }
    }

    fn parse_let_stmt(&mut self) -> Option<Stmt> {
        if !self.expect_peek(Token::Ident(String::from("variable_name"))) {
            return None;
        };

        let name = match self.parse_ident() {
            Some(name) => name,
            None => return None,
        };

        if !self.expect_peek(Token::Assign) {
            return None;
        }

        self.consume();

        let expr = match self.parse_expr(Precedence::PLowest) {
            Some(expr) => expr,
            None => return None,
        };

        if self.peek_token_is(&Token::SemiColon) {
            self.consume();
        }

        Some(Stmt::Let(name, expr))
    }

    fn parse_return_stmt(&mut self) -> Option<Stmt> {
        self.consume();

        let expr = match self.parse_expr(Precedence::PLowest) {
            Some(expr) => expr,
            None => return None,
        };

        if self.peek_token_is(&Token::SemiColon) {
            self.consume();
        }

        Some(Stmt::Return(expr))
    }

    fn parse_expr_stmt(&mut self) -> Option<Stmt> {
        match self.parse_expr(Precedence::PLowest) {
            Some(expr) => {
                if self.peek_token_is(&Token::SemiColon) {
                    self.consume();
                }
                Some(Stmt::Expr(expr))
            }
            None => None,
        }
    }

    fn parse_block_stmt(&mut self) -> BlockStmt {
        self.consume();

        let mut block = vec![];

        while !self.current_token_is(Token::RBrace) && !self.current_token_is(Token::EOF) {
            match self.parse_stmt() {
                Some(stmt) => block.push(stmt),
                None => {}
            }
            self.consume();
        }

        block
    }

    fn parse_expr(&mut self, precedence: Precedence) -> Option<Expr> {
        // prefix
        let mut left = match self.current_token {
            Token::Ident(_) => self.parse_ident_expr(),
            Token::Int(_) => self.parse_int_expr(),
            Token::String(_) => self.parse_string_expr(),
            Token::True => self.parse_bool_expr(),
            Token::False => self.parse_bool_expr(),
            Token::LBracket => self.parse_array_expr(),
            Token::LBrace => self.parse_hash_expr(),
            Token::Bang | Token::Minus | Token::Plus => self.parse_prefix_expr(),
            Token::LParen => self.parse_grouped_expr(),
            Token::If => self.parse_if_expr(),
            Token::Macro => self.parse_macro_expr(),
            Token::Function => self.parse_func_expr(),
            _ => {
                self.error_no_prefix_parse();
                return None;
            }
        };

        // infix
        while !self.peek_token_is(&Token::SemiColon) && precedence < self.peek_precedence() {
            match self.next_token {
                Token::Plus
                | Token::Minus
                | Token::Divide
                | Token::Multiply
                | Token::Equal
                | Token::NotEqual
                | Token::LessThan
                | Token::LessThanEqual
                | Token::GreaterThan
                | Token::GreaterThanEqual => {
                    self.consume();
                    left = self.parse_infix_expr(left.unwrap());
                }
                Token::LBracket => {
                    self.consume();
                    left = self.parse_index_expr(left.unwrap());
                }
                Token::LParen => {
                    self.consume();
                    left = self.parse_call_expr(left.unwrap());
                }
                _ => return left,
            }
        }

        left
    }

    fn parse_ident_expr(&mut self) -> Option<Expr> {
        match self.parse_ident() {
            Some(ident) => Some(Expr::Ident(ident)),
            _ => None,
        }
    }

    fn parse_int_expr(&mut self) -> Option<Expr> {
        match self.current_token {
            Token::Int(ref mut int) => Some(Expr::Lit(Literal::Int(int.clone()))),
            _ => None,
        }
    }

    fn parse_string_expr(&mut self) -> Option<Expr> {
        match self.current_token {
            Token::String(ref mut s) => Some(Expr::Lit(Literal::String(s.clone()))),
            _ => None,
        }
    }

    fn parse_prefix_expr(&mut self) -> Option<Expr> {
        let prefix = match self.current_token {
            Token::Bang => Prefix::Not,
            Token::Minus => Prefix::PreMinus,
            Token::Plus => Prefix::PrePlus,
            _ => return None,
        };

        self.consume();

        match self.parse_expr(Precedence::PPrefix) {
            Some(expr) => Some(Expr::Prefix(prefix, Box::new(expr))),
            None => None,
        }
    }

    fn parse_infix_expr(&mut self, left: Expr) -> Option<Expr> {
        let infix = match self.current_token {
            Token::Plus => Infix::Plus,
            Token::Minus => Infix::Minus,
            Token::Divide => Infix::Divide,
            Token::Multiply => Infix::Multiply,
            Token::Equal => Infix::Equal,
            Token::NotEqual => Infix::NotEqual,
            Token::LessThan => Infix::LessThan,
            Token::LessThanEqual => Infix::LessThanEqual,
            Token::GreaterThan => Infix::GreaterThan,
            Token::GreaterThanEqual => Infix::GreaterThanEqual,
            _ => return None,
        };

        let precedence = self.current_precedence();

        self.consume();

        match self.parse_expr(precedence) {
            Some(expr) => Some(Expr::Infix(infix, Box::new(left), Box::new(expr))),
            None => None,
        }
    }

    fn parse_bool_expr(&mut self) -> Option<Expr> {
        match self.current_token {
            Token::True => Some(Expr::Lit(Literal::Bool(true))),
            Token::False => Some(Expr::Lit(Literal::Bool(false))),
            _ => None,
        }
    }

    fn parse_grouped_expr(&mut self) -> Option<Expr> {
        self.consume();

        let expr = self.parse_expr(Precedence::PLowest);

        if !self.expect_peek(Token::RParen) {
            None
        } else {
            expr
        }
    }

    fn parse_if_expr(&mut self) -> Option<Expr> {
        if !self.expect_peek(Token::LParen) {
            return None;
        }

        self.consume();

        let cond = match self.parse_expr(Precedence::PLowest) {
            Some(expr) => expr,
            None => return None,
        };

        if !self.expect_peek(Token::RParen) {
            return None;
        }

        let consequence = if self.peek_token_is(&Token::LBrace) {
            self.consume();
            self.parse_block_stmt()
        } else {
            self.consume();
            match self.parse_expr(Precedence::PLowest) {
                Some(expr) => vec![Stmt::Expr(expr)],
                None => vec![],
            }
        };

        let mut alternative = None;

        if self.peek_token_is(&Token::SemiColon) {
            self.consume();
        }

        if self.current_token_is(Token::Else) || self.peek_token_is(&Token::Else) {
            if self.peek_token_is(&Token::Else) {
                self.consume();
            }

            alternative = if self.peek_token_is(&Token::LBrace) {
                self.consume();
                Some(self.parse_block_stmt())
            } else {
                self.consume();
                match self.parse_expr(Precedence::PLowest) {
                    Some(expr) => Some(vec![Stmt::Expr(expr)]),
                    None => None,
                }
            };
        }

        Some(Expr::If {
            cond: Box::new(cond),
            consequence,
            alternative,
        })
    }

    fn parse_macro_expr(&mut self) -> Option<Expr> {
        if !self.expect_peek(Token::LParen) {
            return None;
        }

        let params = match self.parse_func_params() {
            Some(params) => params,
            None => return None,
        };

        if !self.expect_peek(Token::LBrace) {
            return None;
        }

        Some(Expr::Lit(Literal::Macro(MacroLiteral {
            params,
            body: Box::new(self.parse_block_stmt()),
        })))
    }

    fn parse_func_expr(&mut self) -> Option<Expr> {
        if !self.expect_peek(Token::LParen) {
            return None;
        }

        let params = match self.parse_func_params() {
            Some(params) => params,
            None => return None,
        };

        if !self.expect_peek(Token::LBrace) {
            return None;
        }

        Some(Expr::Function {
            params,
            body: self.parse_block_stmt(),
        })
    }

    fn parse_func_params(&mut self) -> Option<Vec<Identity>> {
        let mut params = vec![];

        if self.peek_token_is(&Token::RParen) {
            self.consume();
            return Some(params);
        }

        self.consume();

        match self.parse_ident() {
            Some(ident) => params.push(ident),
            None => return None,
        };

        while self.peek_token_is(&Token::Comma) {
            self.consume();
            self.consume();

            match self.parse_ident() {
                Some(ident) => params.push(ident),
                None => return None,
            };
        }

        if !self.expect_peek(Token::RParen) {
            return None;
        }

        Some(params)
    }

    fn parse_call_expr(&mut self, func: Expr) -> Option<Expr> {
        let arguments = match self.parse_expr_list(Token::RParen) {
            Some(args) => args,
            None => return None,
        };

        let name = match &func {
            Expr::Ident(Identity(v)) => Some(v.clone()),
            _ => None,
        };

        Some(Expr::Call {
            function: Box::new(func),
            name,
            arguments,
        })
    }

    fn parse_expr_list(&mut self, end: Token) -> Option<Vec<Expr>> {
        let mut list = vec![];

        if self.peek_token_is(&end) {
            self.consume();
            return Some(list);
        }

        self.consume();

        match self.parse_expr(Precedence::PLowest) {
            Some(expr) => list.push(expr),
            None => return None,
        }

        while self.peek_token_is(&Token::Comma) {
            self.consume();
            self.consume();

            match self.parse_expr(Precedence::PLowest) {
                Some(expr) => list.push(expr),
                None => return None,
            }
        }

        if !self.expect_peek(end) {
            return None;
        }

        Some(list)
    }

    fn parse_array_expr(&mut self) -> Option<Expr> {
        match self.parse_expr_list(Token::RBracket) {
            Some(list) => Some(Expr::Lit(Literal::Array(list))),
            None => None,
        }
    }

    fn parse_hash_expr(&mut self) -> Option<Expr> {
        let mut pairs = Vec::new();

        while !self.peek_token_is(&Token::RBrace) {
            self.consume();

            let key = match self.parse_expr(Precedence::PLowest) {
                Some(expr) => expr,
                None => return None,
            };

            if !self.expect_peek(Token::Colon) {
                return None;
            }

            self.consume();

            let value = match self.parse_expr(Precedence::PLowest) {
                Some(expr) => expr,
                None => return None,
            };

            pairs.push((key, value));

            if !self.peek_token_is(&Token::RBrace) && !self.expect_peek(Token::Comma) {
                return None;
            }
        }

        if !self.expect_peek(Token::RBrace) {
            return None;
        }

        Some(Expr::Lit(Literal::Hash(pairs)))
    }

    fn parse_index_expr(&mut self, left: Expr) -> Option<Expr> {
        self.consume();

        let index = match self.parse_expr(Precedence::PLowest) {
            Some(expr) => expr,
            None => return None,
        };

        if !self.expect_peek(Token::RBracket) {
            return None;
        }

        Some(Expr::Index(Box::new(left), Box::new(index)))
    }
}

#[cfg(test)]
mod tests {
    use super::ast::*;
    use super::Parser;
    use crate::lexer::Lexer;

    macro_rules! either {
        ($test:expr => $true_expr:expr; $false_expr:expr) => {
            if $test {
                $true_expr
            } else {
                $false_expr
            }
        };
    }

    fn check_parser_errors(parser: &mut Parser) {
        let errors = parser.errors();

        if errors.len() == 0 {
            return;
        }

        println!(
            "parser has {} error{}",
            errors.len(),
            either!(errors.len() == 1 => ""; "s")
        );

        for err in errors {
            println!("parser error: [{}] {}", err.kind, err.description);
        }

        panic!("Build failed!");
    }

    fn assert_eq(input: &str, expected: Program) {
        let lexer = Lexer::new(input);
        let mut parser = Parser::new(lexer);
        let result = parser.parse();
        check_parser_errors(&mut parser);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_let_stmt() {
        let input = r#"
      let x = 5;
      let y = 10;
      let foobar = 838383;
    "#;

        let program: Program = vec![
            Stmt::Let(Identity("x".to_owned()), Expr::Lit(Literal::Int(5))),
            Stmt::Let(Identity("y".to_owned()), Expr::Lit(Literal::Int(10))),
            Stmt::Let(
                Identity("foobar".to_owned()),
                Expr::Lit(Literal::Int(838383)),
            ),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_return_stmt() {
        let input = r#"
      return 5;
      return 10;
      return 993322;
    "#;

        let program: Program = vec![
            Stmt::Return(Expr::Lit(Literal::Int(5))),
            Stmt::Return(Expr::Lit(Literal::Int(10))),
            Stmt::Return(Expr::Lit(Literal::Int(993322))),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_identity_expr() {
        let input = "foobar;";

        let program: Program = vec![Stmt::Expr(Expr::Ident(Identity(String::from("foobar"))))];

        assert_eq(input, program);
    }

    #[test]
    fn test_integer_expr() {
        let input = "5;";

        let program: Program = vec![Stmt::Expr(Expr::Lit(Literal::Int(5)))];

        assert_eq(input, program);
    }

    #[test]
    fn test_boolean_literal_expr() {
        let input = r#"
      true;
      false;
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::Lit(Literal::Bool(true))),
            Stmt::Expr(Expr::Lit(Literal::Bool(false))),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_string_literal_expr() {
        let input = "\"hello world\";";

        let program: Program = vec![Stmt::Expr(Expr::Lit(Literal::String(String::from(
            "hello world",
        ))))];

        assert_eq(input, program);
    }

    #[test]
    fn test_array_literal_expr() {
        let input = "[1, 2 * 2, 3 + 3]";

        let program: Program = vec![Stmt::Expr(Expr::Lit(Literal::Array(vec![
            Expr::Lit(Literal::Int(1)),
            Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Lit(Literal::Int(2))),
                Box::new(Expr::Lit(Literal::Int(2))),
            ),
            Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Lit(Literal::Int(3))),
                Box::new(Expr::Lit(Literal::Int(3))),
            ),
        ])))];

        assert_eq(input, program);
    }

    #[test]
    fn test_hash_literal_expr() {
        let input = r#"
      {};
      {"one": 1, "two": 2, "three": 3};
      {"one": 0 + 1, "two": 10 - 8, "three": 15 / 5};
      {key: "value"};
    "#;
        let program: Program = vec![
            Stmt::Expr(Expr::Lit(Literal::Hash(vec![]))),
            Stmt::Expr(Expr::Lit(Literal::Hash(vec![
                (
                    Expr::Lit(Literal::String(String::from("one"))),
                    Expr::Lit(Literal::Int(1)),
                ),
                (
                    Expr::Lit(Literal::String(String::from("two"))),
                    Expr::Lit(Literal::Int(2)),
                ),
                (
                    Expr::Lit(Literal::String(String::from("three"))),
                    Expr::Lit(Literal::Int(3)),
                ),
            ]))),
            Stmt::Expr(Expr::Lit(Literal::Hash(vec![
                (
                    Expr::Lit(Literal::String(String::from("one"))),
                    Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Lit(Literal::Int(0))),
                        Box::new(Expr::Lit(Literal::Int(1))),
                    ),
                ),
                (
                    Expr::Lit(Literal::String(String::from("two"))),
                    Expr::Infix(
                        Infix::Minus,
                        Box::new(Expr::Lit(Literal::Int(10))),
                        Box::new(Expr::Lit(Literal::Int(8))),
                    ),
                ),
                (
                    Expr::Lit(Literal::String(String::from("three"))),
                    Expr::Infix(
                        Infix::Divide,
                        Box::new(Expr::Lit(Literal::Int(15))),
                        Box::new(Expr::Lit(Literal::Int(5))),
                    ),
                ),
            ]))),
            Stmt::Expr(Expr::Lit(Literal::Hash(vec![(
                Expr::Ident(Identity(String::from("key"))),
                Expr::Lit(Literal::String(String::from("value"))),
            )]))),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_macro_literal_parse() {
        let input = "macro(x, y) { x + y; }";

        let l = Lexer::new(input);
        let mut p = Parser::new(l);
        let program = p.parse();
        assert_eq!(program.len(), 1);

        if let Stmt::Expr(stmt) = &program[0] {
            if let Expr::Lit(Literal::Macro(MacroLiteral { params, body })) = &stmt {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0], Identity("x".to_string()));
                assert_eq!(params[1], Identity("y".to_string()));
                assert_eq!(body.len(), 1);
                if let Stmt::Expr(body_stmt) = &body[0] {
                    println!("{:?}", body_stmt);
                } else {
                    assert!(false, "macro body stmt is not ast::Expr")
                }
            } else {
                assert!(false, "stmt is not ast::MacroLiteral")
            }
        } else {
            assert!(false, "program[0] is not ast::Expr")
        }
    }

    #[test]
    fn test_index_expr() {
        let input = "myArray[1 + 1]";

        let program: Program = vec![Stmt::Expr(Expr::Index(
            Box::new(Expr::Ident(Identity(String::from("myArray")))),
            Box::new(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Lit(Literal::Int(1))),
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        ))];

        assert_eq(input, program);
    }

    #[test]
    fn test_prefix_exprs() {
        let input = r#"
      !5;
      -15;
      +15;
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::Prefix(
                Prefix::Not,
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Prefix(
                Prefix::PreMinus,
                Box::new(Expr::Lit(Literal::Int(15))),
            )),
            Stmt::Expr(Expr::Prefix(
                Prefix::PrePlus,
                Box::new(Expr::Lit(Literal::Int(15))),
            )),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_infix_exprs() {
        let input = r#"
      5 + 5;
      5 - 5;
      5 * 5;
      5 / 5;
      5 > 5;
      5 < 5;
      5 == 5;
      5 != 5;
      5 >= 5;
      5 <= 5;
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Minus,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Divide,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::GreaterThan,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::LessThan,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::NotEqual,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::GreaterThanEqual,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::LessThanEqual,
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_if_expr() {
        let input = r#"
      if (x < y) { x };
      if (x < y) x;
      if (x < y) x; y;
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: None,
            }),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: None,
            }),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: None,
            }),
            Stmt::Expr(Expr::Ident(Identity(String::from("y")))),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_if_else_expr() {
        let input = r#"
      if (x < y) { x } else { y }
      if (x < y) x; else y; x;
      if (x < y) x else y; x;
      if (x < y) { x; } else y; x;
      if (x < y) x; else { y; }
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: Some(vec![Stmt::Expr(Expr::Ident(Identity(String::from("y"))))]),
            }),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: Some(vec![Stmt::Expr(Expr::Ident(Identity(String::from("y"))))]),
            }),
            Stmt::Expr(Expr::Ident(Identity(String::from("x")))),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: Some(vec![Stmt::Expr(Expr::Ident(Identity(String::from("y"))))]),
            }),
            Stmt::Expr(Expr::Ident(Identity(String::from("x")))),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: Some(vec![Stmt::Expr(Expr::Ident(Identity(String::from("y"))))]),
            }),
            Stmt::Expr(Expr::Ident(Identity(String::from("x")))),
            Stmt::Expr(Expr::If {
                cond: Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Ident(Identity(String::from("x")))),
                    Box::new(Expr::Ident(Identity(String::from("y")))),
                )),
                consequence: vec![Stmt::Expr(Expr::Ident(Identity(String::from("x"))))],
                alternative: Some(vec![Stmt::Expr(Expr::Ident(Identity(String::from("y"))))]),
            }),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_func_expr() {
        let input = "fn(x, y) { x + y; }";

        let program: Program = vec![Stmt::Expr(Expr::Function {
            params: vec![Identity(String::from("x")), Identity(String::from("y"))],
            body: vec![Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Ident(Identity(String::from("x")))),
                Box::new(Expr::Ident(Identity(String::from("y")))),
            ))],
        })];

        assert_eq(input, program);
    }

    #[test]
    fn test_func_params() {
        let input = r#"
      fn() {};
      fn(x) {};
      fn(x, y, z) {};
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::Function {
                params: vec![],
                body: vec![],
            }),
            Stmt::Expr(Expr::Function {
                params: vec![Identity(String::from("x"))],
                body: vec![],
            }),
            Stmt::Expr(Expr::Function {
                params: vec![
                    Identity(String::from("x")),
                    Identity(String::from("y")),
                    Identity(String::from("z")),
                ],
                body: vec![],
            }),
        ];

        assert_eq(input, program);
    }

    #[test]
    fn test_call_expr() {
        let input = "add(1, 2 * 3, 4 + 5);";

        let program: Program = vec![Stmt::Expr(Expr::Call {
            function: Box::new(Expr::Ident(Identity(String::from("add")))),
            name: Some(String::from("add")),
            arguments: vec![
                Expr::Lit(Literal::Int(1)),
                Expr::Infix(
                    Infix::Multiply,
                    Box::new(Expr::Lit(Literal::Int(2))),
                    Box::new(Expr::Lit(Literal::Int(3))),
                ),
                Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(4))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                ),
            ],
        })];

        assert_eq(input, program);
    }

    #[test]
    fn test_operator_precedence_parsing() {
        let input = r#"
      -a * b;
      !-a;
      a + b + c;
      a + b - c;
      a * b * c;
      a * b / c;
      a + b / c;
      a + b * c + d / e - f;
      3 + 4; -5 * 5;
      5 > 4 == 3 < 4;
      5 < 4 != 3 > 4;
      5 >= 4 == 3 <= 4;
      5 <= 4 != 3 >= 4;
      3 + 4 * 5 == 3 * 1 + 4 * 5;
      true;
      false;
      3 > 5 == false;
      3 < 5 == true;
      1 + (2 + 3) + 4;
      (5 + 5) * 2;
      2 / (5 + 5);
      -(5 + 5);
      !(true == true);
      a + add(b * c) + d;
      add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8));
      add(a + b + c * d / f + g);
      a * [1, 2, 3, 4][b * c] * d;
      add(a * b[2], b[1], 2 * [1, 2][1]);
    "#;

        let input2 = r#"
      ((-a) * b);
      (!(-a));
      ((a + b) + c);
      ((a + b) - c);
      ((a * b) * c);
      ((a * b) / c);
      (a + (b / c));
      (((a + (b * c)) + (d / e)) - f);
      (3 + 4); ((-5) * 5);
      (5 > 4) == (3 < 4);
      5 < 4 != 3 > 4;
      5 >= 4 == 3 <= 4;
      5 <= 4 != 3 >= 4;
      3 + 4 * 5 == 3 * 1 + 4 * 5;
      true;
      false;
      3 > 5 == false;
      3 < 5 == true;
      1 + (2 + 3) + 4;
      (5 + 5) * 2;
      2 / (5 + 5);
      -(5 + 5);
      !(true == true);
      (a + add(b * c) + d);
      (add(a, b, 1, 2 * 3, 4 + 5, add(6, 7 * 8)));
      (add(a + b + c * d / f + g));
      (a * [1, 2, 3, 4][b * c] * d);
      (add(a * b[2], b[1], 2 * [1, 2][1]));
    "#;

        let program: Program = vec![
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Prefix(
                    Prefix::PreMinus,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                )),
                Box::new(Expr::Ident(Identity(String::from("b")))),
            )),
            Stmt::Expr(Expr::Prefix(
                Prefix::Not,
                Box::new(Expr::Prefix(
                    Prefix::PreMinus,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Ident(Identity(String::from("b")))),
                )),
                Box::new(Expr::Ident(Identity(String::from("c")))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Minus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Ident(Identity(String::from("b")))),
                )),
                Box::new(Expr::Ident(Identity(String::from("c")))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Infix(
                    Infix::Multiply,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Ident(Identity(String::from("b")))),
                )),
                Box::new(Expr::Ident(Identity(String::from("c")))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Divide,
                Box::new(Expr::Infix(
                    Infix::Multiply,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Ident(Identity(String::from("b")))),
                )),
                Box::new(Expr::Ident(Identity(String::from("c")))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Ident(Identity(String::from("a")))),
                Box::new(Expr::Infix(
                    Infix::Divide,
                    Box::new(Expr::Ident(Identity(String::from("b")))),
                    Box::new(Expr::Ident(Identity(String::from("c")))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Minus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Ident(Identity(String::from("a")))),
                        Box::new(Expr::Infix(
                            Infix::Multiply,
                            Box::new(Expr::Ident(Identity(String::from("b")))),
                            Box::new(Expr::Ident(Identity(String::from("c")))),
                        )),
                    )),
                    Box::new(Expr::Infix(
                        Infix::Divide,
                        Box::new(Expr::Ident(Identity(String::from("d")))),
                        Box::new(Expr::Ident(Identity(String::from("e")))),
                    )),
                )),
                Box::new(Expr::Ident(Identity(String::from("f")))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Lit(Literal::Int(3))),
                Box::new(Expr::Lit(Literal::Int(4))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Prefix(
                    Prefix::PreMinus,
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Infix(
                    Infix::GreaterThan,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
                Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::NotEqual,
                Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
                Box::new(Expr::Infix(
                    Infix::GreaterThan,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Infix(
                    Infix::GreaterThanEqual,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
                Box::new(Expr::Infix(
                    Infix::LessThanEqual,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::NotEqual,
                Box::new(Expr::Infix(
                    Infix::LessThanEqual,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
                Box::new(Expr::Infix(
                    Infix::GreaterThanEqual,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(4))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Lit(Literal::Int(4))),
                        Box::new(Expr::Lit(Literal::Int(5))),
                    )),
                )),
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Lit(Literal::Int(3))),
                        Box::new(Expr::Lit(Literal::Int(1))),
                    )),
                    Box::new(Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Lit(Literal::Int(4))),
                        Box::new(Expr::Lit(Literal::Int(5))),
                    )),
                )),
            )),
            Stmt::Expr(Expr::Lit(Literal::Bool(true))),
            Stmt::Expr(Expr::Lit(Literal::Bool(false))),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Infix(
                    Infix::GreaterThan,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
                Box::new(Expr::Lit(Literal::Bool(false))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Equal,
                Box::new(Expr::Infix(
                    Infix::LessThan,
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
                Box::new(Expr::Lit(Literal::Bool(true))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(1))),
                    Box::new(Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Lit(Literal::Int(2))),
                        Box::new(Expr::Lit(Literal::Int(3))),
                    )),
                )),
                Box::new(Expr::Lit(Literal::Int(4))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
                Box::new(Expr::Lit(Literal::Int(2))),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Divide,
                Box::new(Expr::Lit(Literal::Int(2))),
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
            )),
            Stmt::Expr(Expr::Prefix(
                Prefix::PreMinus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Lit(Literal::Int(5))),
                    Box::new(Expr::Lit(Literal::Int(5))),
                )),
            )),
            Stmt::Expr(Expr::Prefix(
                Prefix::Not,
                Box::new(Expr::Infix(
                    Infix::Equal,
                    Box::new(Expr::Lit(Literal::Bool(true))),
                    Box::new(Expr::Lit(Literal::Bool(true))),
                )),
            )),
            Stmt::Expr(Expr::Infix(
                Infix::Plus,
                Box::new(Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Call {
                        function: Box::new(Expr::Ident(Identity(String::from("add")))),
                        name: Some(String::from("add")),
                        arguments: vec![Expr::Infix(
                            Infix::Multiply,
                            Box::new(Expr::Ident(Identity(String::from("b")))),
                            Box::new(Expr::Ident(Identity(String::from("c")))),
                        )],
                    }),
                )),
                Box::new(Expr::Ident(Identity(String::from("d")))),
            )),
            Stmt::Expr(Expr::Call {
                function: Box::new(Expr::Ident(Identity(String::from("add")))),
                name: Some(String::from("add")),
                arguments: vec![
                    Expr::Ident(Identity(String::from("a"))),
                    Expr::Ident(Identity(String::from("b"))),
                    Expr::Lit(Literal::Int(1)),
                    Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Lit(Literal::Int(2))),
                        Box::new(Expr::Lit(Literal::Int(3))),
                    ),
                    Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Lit(Literal::Int(4))),
                        Box::new(Expr::Lit(Literal::Int(5))),
                    ),
                    Expr::Call {
                        function: Box::new(Expr::Ident(Identity(String::from("add")))),
                        name: Some(String::from("add")),
                        arguments: vec![
                            Expr::Lit(Literal::Int(6)),
                            Expr::Infix(
                                Infix::Multiply,
                                Box::new(Expr::Lit(Literal::Int(7))),
                                Box::new(Expr::Lit(Literal::Int(8))),
                            ),
                        ],
                    },
                ],
            }),
            Stmt::Expr(Expr::Call {
                function: Box::new(Expr::Ident(Identity(String::from("add")))),
                name: Some(String::from("add")),
                arguments: vec![Expr::Infix(
                    Infix::Plus,
                    Box::new(Expr::Infix(
                        Infix::Plus,
                        Box::new(Expr::Infix(
                            Infix::Plus,
                            Box::new(Expr::Ident(Identity(String::from("a")))),
                            Box::new(Expr::Ident(Identity(String::from("b")))),
                        )),
                        Box::new(Expr::Infix(
                            Infix::Divide,
                            Box::new(Expr::Infix(
                                Infix::Multiply,
                                Box::new(Expr::Ident(Identity(String::from("c")))),
                                Box::new(Expr::Ident(Identity(String::from("d")))),
                            )),
                            Box::new(Expr::Ident(Identity(String::from("f")))),
                        )),
                    )),
                    Box::new(Expr::Ident(Identity(String::from("g")))),
                )],
            }),
            Stmt::Expr(Expr::Infix(
                Infix::Multiply,
                Box::new(Expr::Infix(
                    Infix::Multiply,
                    Box::new(Expr::Ident(Identity(String::from("a")))),
                    Box::new(Expr::Index(
                        Box::new(Expr::Lit(Literal::Array(vec![
                            Expr::Lit(Literal::Int(1)),
                            Expr::Lit(Literal::Int(2)),
                            Expr::Lit(Literal::Int(3)),
                            Expr::Lit(Literal::Int(4)),
                        ]))),
                        Box::new(Expr::Infix(
                            Infix::Multiply,
                            Box::new(Expr::Ident(Identity(String::from("b")))),
                            Box::new(Expr::Ident(Identity(String::from("c")))),
                        )),
                    )),
                )),
                Box::new(Expr::Ident(Identity(String::from("d")))),
            )),
            Stmt::Expr(Expr::Call {
                function: Box::new(Expr::Ident(Identity(String::from("add")))),
                name: Some(String::from("add")),
                arguments: vec![
                    Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Ident(Identity(String::from("a")))),
                        Box::new(Expr::Index(
                            Box::new(Expr::Ident(Identity(String::from("b")))),
                            Box::new(Expr::Lit(Literal::Int(2))),
                        )),
                    ),
                    Expr::Index(
                        Box::new(Expr::Ident(Identity(String::from("b")))),
                        Box::new(Expr::Lit(Literal::Int(1))),
                    ),
                    Expr::Infix(
                        Infix::Multiply,
                        Box::new(Expr::Lit(Literal::Int(2))),
                        Box::new(Expr::Index(
                            Box::new(Expr::Lit(Literal::Array(vec![
                                Expr::Lit(Literal::Int(1)),
                                Expr::Lit(Literal::Int(2)),
                            ]))),
                            Box::new(Expr::Lit(Literal::Int(1))),
                        )),
                    ),
                ],
            }),
        ];

        assert_eq(input, program);

        let lexer1 = Lexer::new(input);
        let mut parser1 = Parser::new(lexer1);
        let result1 = parser1.parse();
        check_parser_errors(&mut parser1);

        let lexer2 = Lexer::new(input2);
        let mut parser2 = Parser::new(lexer2);
        let result2 = parser2.parse();
        check_parser_errors(&mut parser2);

        assert_eq!(result1, result2);
    }
}
