use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Instructions(pub Vec<u8>);

impl std::ops::Deref for Instructions {
    type Target = Vec<u8>;
    
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Instructions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for Instructions {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;
    
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub type Opcode = u8;

pub const OPCONSTANT: Opcode = 0;
pub const OPADD: Opcode = 1;
pub const OPSUB: Opcode = 2;
pub const OPMULT: Opcode = 3;
pub const OPDIV: Opcode = 4;
pub const OPTRUE: Opcode = 5;
pub const OPFALSE: Opcode = 6;
pub const OPGREATERTHAN: Opcode = 7;
pub const OPGREATERTHANEQUAL: Opcode = 8;
pub const OPLESSTHAN: Opcode = 9;
pub const OPLESSTHANEQUAL: Opcode = 10;
pub const OPEQUAL: Opcode = 11;
pub const OPNOTEQUAL: Opcode = 12;
pub const OPMINUS: Opcode = 13;
pub const OPPLUS: Opcode = 14;
pub const OPBANG: Opcode = 15;
pub const OPPOP: Opcode = 16;
pub const OPJUMPELSE: Opcode = 17;
pub const OPJUMP: Opcode = 18;
pub const OPNULL: Opcode = 19;
pub const OPGETGLOBAL: Opcode = 20;
pub const OPSETGLOBAL: Opcode = 21;
pub const OPARRAY: Opcode = 22;
pub const OPHASH: Opcode = 23;
pub const OPINDEX: Opcode = 24;
pub const OPCALL: Opcode = 25;
pub const OPRETURNVALUE: Opcode = 26;
pub const OPRETURN: Opcode = 27;
pub const OPGETLOCAL: Opcode = 28;
pub const OPSETLOCAL: Opcode = 29;

pub struct Definition {
    pub name: String,
    pub operand_widths: Vec<usize>,
}

use once_cell::sync::Lazy;
static DEFINITIONS: Lazy<std::collections::HashMap<Opcode, Definition>> = Lazy::new(|| {
    let mut m = std::collections::HashMap::new();
    m.insert(OPCONSTANT, Definition {
        name: "OpConstant".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPADD, Definition {
        name: "OpAdd".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPSUB, Definition {
        name: "OpSub".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPMULT, Definition {
        name: "OpMult".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPDIV, Definition {
        name: "OpDiv".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPTRUE, Definition {
        name: "OpTrue".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPFALSE, Definition {
        name: "OpFalse".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPGREATERTHAN, Definition {
        name: "OpGreaterThan".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPGREATERTHANEQUAL, Definition {
        name: "OpGreaterThanEqual".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPLESSTHAN, Definition {
        name: "OpLessThan".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPLESSTHANEQUAL, Definition {
        name: "OpLessThanEqual".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPEQUAL, Definition {
        name: "OpEqual".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPNOTEQUAL, Definition {
        name: "OpNotEqual".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPMINUS, Definition {
        name: "OpMinus".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPPLUS, Definition {
        name: "OpPlus".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPBANG, Definition {
        name: "OpBang".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPPOP, Definition {
        name: "OpPop".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPJUMPELSE, Definition {
        name: "OpJumpElse".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPJUMP, Definition {
        name: "OpJump".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPNULL, Definition {
        name: "OpNull".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPGETGLOBAL, Definition {
        name: "OpGetGlobal".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPSETGLOBAL, Definition {
        name: "OpSetGlobal".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPARRAY, Definition {
        name: "OpArray".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPHASH, Definition {
        name: "OpHash".to_string(),
        operand_widths: vec![2],
    });
    m.insert(OPINDEX, Definition {
        name: "OpIndex".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPCALL, Definition {
        name: "OpCall".to_string(),
        operand_widths: vec![1],
    });
    m.insert(OPRETURNVALUE, Definition {
        name: "OpReturnValue".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPRETURN, Definition {
        name: "OpReturn".to_string(),
        operand_widths: vec![],
    });
    m.insert(OPGETLOCAL, Definition {
        name: "OpGetLocal".to_string(),
        operand_widths: vec![1],
    });
    m.insert(OPSETLOCAL, Definition {
        name: "OpSetLocal".to_string(),
        operand_widths: vec![1],
    });
    m
});

impl Instructions {
    pub fn new() -> Self {
        Instructions(Vec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, index: usize) -> Option<u8> {
        self.0.get(index).copied()
    }

    pub fn fmt_instruction(def: &Definition, operands: &[i32]) -> String {
        let operand_count = def.operand_widths.len();

        if operands.len() != operand_count {
            return format!(
                "ERROR: operand len {} does not match defined {}\n",
                operands.len(),
                operand_count
            );
        }

        match operand_count {
            0 => def.name.clone(),
            1 => format!("{} {}", def.name, operands[0]),
            _ => format!(
                "ERROR: unhandled operandCount for {}\n",
                def.name
            ),
        }
    }
}

impl fmt::Display for Instructions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut i = 0;
        while i < self.len() {
            let op = self.get(i).unwrap_or(0);
            let def = match lookup(op) {
                Ok(def) => def,
                Err(err) => {
                    writeln!(f, "ERROR: {}", err)?;
                    i += 1;
                    continue;
                }
            };

            let (operands, read) = read_operands(def, &self.0[i + 1..]);
            writeln!(
                f,
                "{:04} {}",
                i,
                Instructions::fmt_instruction(def, &operands)
            )?;
            i += 1 + read;
        }
        Ok(())
    }
}


pub fn lookup(op: u8) -> Result<&'static Definition, String> {
    match DEFINITIONS.get(&op) {
        Some(def) => Ok(def),
        None => Err(format!("opcode {} undefined", op)),
    }
}

pub fn make(op: Opcode, operands: &[i32]) -> Instructions {
    let def = match lookup(op) {
        Ok(def) => def,
        Err(_) => return Instructions(vec![]),
    };

    let mut instruction_len = 1;
    for w in &def.operand_widths {
        instruction_len += w;
    }

    let mut instruction = vec![0u8; instruction_len];
    instruction[0] = op as u8;

    let mut offset = 1;
    for (i, o) in operands.iter().enumerate() {
        let width = def.operand_widths[i];
        match width {
            2 => {
                let val = *o as u16;
                instruction[offset] = (val >> 8) as u8;
                instruction[offset + 1] = (val & 0xff) as u8;
            }
            1 => {
                instruction[offset] = *o as u8;
            }
            _ => {}
        }
        offset += width;
    }

    Instructions(instruction)
}

pub fn read_operands(def: &Definition, ins: &[u8]) -> (Vec<i32>, usize) {
    let mut operands = Vec::with_capacity(def.operand_widths.len());
    let mut offset = 0;

    for width in &def.operand_widths {
        match *width {
            2 => {
                let val = read_u16(&ins[offset..offset + 2]);
                operands.push(val as i32);
            }
            1 => {
                let val = read_u8(&ins[offset]);
                operands.push(val);
            }
            _ => {}
        }
        offset += *width;
    }

    (operands, offset)
}

pub fn read_u8(ins: &u8) -> i32 {
    *ins as i32
}

pub fn read_u16(ins: &[u8]) -> u16 {
    ((ins[0] as u16) << 8) | (ins[1] as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make() {
        struct TestCase {
            op: Opcode,
            operands: Vec<i32>,
            expected: Vec<u8>,
        }

        let tests = vec![
            TestCase {
                op: OPCONSTANT,
                operands: vec![65534],
                expected: vec![OPCONSTANT, 255, 254],
            },
            TestCase {
                op: OPGETLOCAL,
                operands: vec![255],
                expected: vec![OPGETLOCAL, 255],
            },
            TestCase {
                op: OPSETLOCAL,
                operands: vec![255],
                expected: vec![OPSETLOCAL, 255],
            },
        ];

        for tt in tests {
            let instruction = make(tt.op, &tt.operands);

            assert_eq!(
                instruction.len(),
                tt.expected.len(),
                "instruction has wrong length. want={}, got={}",
                tt.expected.len(),
                instruction.len()
            );

            for (i, b) in tt.expected.iter().enumerate() {
                assert_eq!(
                    instruction.get(i).unwrap(), *b,
                    "wrong byte at pos {}. want={}, got={}",
                    i, b, instruction.get(i).unwrap()
                );
            }
        }
    }

    #[test]
    fn test_instructions_string() {
        // Prepare instructions
        let instructions = vec![
            make(OPADD, &[]),
            make(OPGETLOCAL, &[1]),
            make(OPCONSTANT, &[2]),
            make(OPCONSTANT, &[65535]),
        ];

        // Concatenate all instructions into a single Vec<u8>
        let mut concatted = vec![];
        for ins in instructions {
            concatted.extend(ins);
        }

        // Expected string
        let expected = "0000 OpAdd\n0001 OpGetLocal 1\n0003 OpConstant 2\n0006 OpConstant 65535\n";

        // Use the Instructions::to_string() or similar method
        let got = instructions_to_string(&concatted);

        assert_eq!(
            got, expected,
            "instructions wrongly formatted.\nwant={:?}\ngot={:?}",
            expected, got
        );
    }

    #[test]
    fn test_read_operands() {
        struct TestCase {
            op: Opcode,
            operands: Vec<i32>,
            bytes_read: usize,
        }

        let tests = vec![
            TestCase {
                op: OPCONSTANT,
                operands: vec![65535],
                bytes_read: 2,
            },
            TestCase {
                op: OPGETLOCAL,
                operands: vec![255],
                bytes_read: 1,
            },
        ];

        for tt in tests {
            let instruction = make(tt.op, &tt.operands);

            let def = lookup(tt.op).expect("definition not found");

            let (operands_read, n) = read_operands(def, &instruction.0[1..]);
            assert_eq!(
                n, tt.bytes_read,
                "n wrong. want={}, got={}",
                tt.bytes_read, n
            );

            for (i, want) in tt.operands.iter().enumerate() {
                assert_eq!(
                    operands_read[i], *want,
                    "operand wrong. want={}, got={}",
                    want, operands_read[i]
                );
            }
        }
    }

    // Helper function to format instructions as string, similar to Go's Instructions.String()
    fn instructions_to_string(ins: &[u8]) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < ins.len() {
            let op = ins[i];
            match lookup(op) {
                Ok(def) => {
                    let (operands, read) = read_operands(def, &ins[i + 1..]);
                    // Write offset
                    out.push_str(&format!("{:04} {}", i, def.name));
                    for operand in operands {
                        out.push_str(&format!(" {}", operand));
                    }
                    out.push('\n');
                    i += 1 + read;
                }
                Err(_) => {
                    // Unknown opcode, just print it
                    out.push_str(&format!("{:04} UNKNOWN\n", i));
                    i += 1;
                }
            }
        }
        out
    }
}