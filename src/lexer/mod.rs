use crate::token::Token;

pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    read_position: usize,
    ch: u8,
}

impl<'a> Lexer<'a> {
    /**
     * public functions
     **/

    pub fn new(input: &'a str) -> Self {
        let mut l = Lexer {
            input,
            position: 0,
            read_position: 0,
            ch: 0,
        };

        l.read_char();

        return l;
    }

    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let tok: Token = match self.ch {
            b'=' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::Equal
                } else {
                    Token::Assign
                }
            }
            b'+' => Token::Plus,
            b'-' => Token::Minus,
            b'!' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::NotEqual
                } else {
                    Token::Bang
                }
            }
            b'*' => Token::Multiply,
            b'/' => Token::Divide,
            b'<' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::LessThanEqual
                } else {
                    Token::LessThan
                }
            }
            b'>' => {
                if self.peek_char() == b'=' {
                    self.read_char();
                    Token::GreaterThanEqual
                } else {
                    Token::GreaterThan
                }
            }
            b'(' => Token::LParen,
            b')' => Token::RParen,
            b',' => Token::Comma,
            b'{' => Token::LBrace,
            b'}' => Token::RBrace,
            b'[' => Token::LBracket,
            b']' => Token::RBracket,
            b':' => Token::Colon,
            b';' => Token::SemiColon,
            b'"' => self.eat_string(),
            b'0'..=b'9' => self.eat_number(),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.eat_identifier(),
            0 => Token::EOF,
            _ => Token::Illegal,
        };

        self.read_char();

        return tok;
    }

    /**
     * private functions
     **/

    fn read_char(&mut self) {
        if self.read_position >= self.input.len() {
            self.ch = 0;
        } else {
            self.ch = self.input.as_bytes()[self.read_position];
        }

        self.position = self.read_position;
        self.read_position += 1;
    }

    fn peek_char(&mut self) -> u8 {
        if self.read_position >= self.input.len() {
            return 0;
        } else {
            return self.input.as_bytes()[self.read_position];
        }
    }

    fn put_back(&mut self) {
        self.read_position -= 1;
    }

    fn skip_whitespace(&mut self) {
        loop {
            match self.ch {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }
    }

    fn eat_string(&mut self) -> Token {
        self.read_char();

        let first_char = self.position;

        loop {
            match self.ch {
                b'"' => {
                    let str_literal = &self.input[first_char..self.position];
                    return Token::String(str_literal.to_string());
                }
                _ => {
                    self.read_char();
                }
            }
        }
    }

    fn eat_number(&mut self) -> Token {
        let pos = self.position;

        loop {
            match self.ch {
                b'0'..=b'9' => {
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }

        let number = &self.input[pos..self.position];

        self.put_back();

        Token::Int(number.parse::<i64>().unwrap())
    }

    fn eat_identifier(&mut self) -> Token {
        let pos = self.position;

        loop {
            match self.ch {
                b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'$' => {
                    self.read_char();
                }
                _ => {
                    break;
                }
            }
        }

        let kw_or_ident = &self.input[pos..self.position];

        self.put_back();

        match kw_or_ident {
            "fn" => Token::Function,
            "macro" => Token::Macro,
            "let" => Token::Let,
            "true" => Token::True,
            "false" => Token::False,
            "if" => Token::If,
            "else" => Token::Else,
            "return" => Token::Return,
            _ => Token::Ident(String::from(kw_or_ident)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::lexer::Lexer;
    use crate::token::Token;

    #[test]
    fn test_next_token() {
        let input = r#"let five = 5;
let ten = 10;
let add = fn(x, y) {
  x + y;
};
let result = add(five, ten);
!-/*5;
5 < 10 > 5;
if (5 < 10) {
  return true;
} else {
  return false;
}
10 == 10;
10 != 9;
10 <= 10;
10 >= 10;
"foobar";
"foo bar";
[1, 2];
{"foo": "bar"};
"#;

        let tests = vec![
            Token::Let,
            Token::Ident(String::from("five")),
            Token::Assign,
            Token::Int(5),
            Token::SemiColon,
            Token::Let,
            Token::Ident(String::from("ten")),
            Token::Assign,
            Token::Int(10),
            Token::SemiColon,
            Token::Let,
            Token::Ident(String::from("add")),
            Token::Assign,
            Token::Function,
            Token::LParen,
            Token::Ident(String::from("x")),
            Token::Comma,
            Token::Ident(String::from("y")),
            Token::RParen,
            Token::LBrace,
            Token::Ident(String::from("x")),
            Token::Plus,
            Token::Ident(String::from("y")),
            Token::SemiColon,
            Token::RBrace,
            Token::SemiColon,
            Token::Let,
            Token::Ident(String::from("result")),
            Token::Assign,
            Token::Ident(String::from("add")),
            Token::LParen,
            Token::Ident(String::from("five")),
            Token::Comma,
            Token::Ident(String::from("ten")),
            Token::RParen,
            Token::SemiColon,
            Token::Bang,
            Token::Minus,
            Token::Divide,
            Token::Multiply,
            Token::Int(5),
            Token::SemiColon,
            Token::Int(5),
            Token::LessThan,
            Token::Int(10),
            Token::GreaterThan,
            Token::Int(5),
            Token::SemiColon,
            Token::If,
            Token::LParen,
            Token::Int(5),
            Token::LessThan,
            Token::Int(10),
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::True,
            Token::SemiColon,
            Token::RBrace,
            Token::Else,
            Token::LBrace,
            Token::Return,
            Token::False,
            Token::SemiColon,
            Token::RBrace,
            Token::Int(10),
            Token::Equal,
            Token::Int(10),
            Token::SemiColon,
            Token::Int(10),
            Token::NotEqual,
            Token::Int(9),
            Token::SemiColon,
            Token::Int(10),
            Token::LessThanEqual,
            Token::Int(10),
            Token::SemiColon,
            Token::Int(10),
            Token::GreaterThanEqual,
            Token::Int(10),
            Token::SemiColon,
            Token::String(String::from("foobar")),
            Token::SemiColon,
            Token::String(String::from("foo bar")),
            Token::SemiColon,
            Token::LBracket,
            Token::Int(1),
            Token::Comma,
            Token::Int(2),
            Token::RBracket,
            Token::SemiColon,
            Token::LBrace,
            Token::String(String::from("foo")),
            Token::Colon,
            Token::String(String::from("bar")),
            Token::RBrace,
            Token::SemiColon,
            Token::EOF,
        ];

        let mut lexer = Lexer::new(input);

        for expect in tests {
            let tok = lexer.next_token();

            println!("{:?}", tok);

            assert_eq!(expect, tok);
        }
    }
}
