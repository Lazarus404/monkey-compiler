use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
    Builtin,
	FreeScope,
	FunctionScope,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolTable {
    pub outer: Option<Rc<RefCell<SymbolTable>>>,
    store: std::collections::HashMap<String, Symbol>,
    pub num_definitions: usize,
    pub free_symbols: Vec<Symbol>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            outer: None,
            store: std::collections::HashMap::new(),
            num_definitions: 0,
            free_symbols: Vec::new(),
        }
    }

    pub fn new_enclosed(outer: Rc<RefCell<SymbolTable>>) -> Self {
        SymbolTable {
            outer: Some(outer),
            store: std::collections::HashMap::new(),
            num_definitions: 0,
            free_symbols: Vec::new(),
        }
    }

    pub fn define(&mut self, name: &str) -> Symbol {
        let scope = if self.outer.is_none() {
            SymbolScope::Global
        } else {
            SymbolScope::Local
        };

        let symbol = Symbol {
            name: name.to_string(),
            scope,
            index: self.num_definitions,
        };

        self.store.insert(name.to_string(), symbol.clone());
        self.num_definitions += 1;
        symbol
    }

    pub fn define_builtin(&mut self, index: usize, name: String) -> Symbol {
        let symbol = Symbol {
            name: name.clone(),
            scope: SymbolScope::Builtin,
            index,
        };

        self.store.insert(name, symbol.clone());
        symbol
    }

    pub fn define_function_name(&mut self, name: &str) -> Symbol {
        let symbol = Symbol {
            name: name.to_string(),
            scope: SymbolScope::FunctionScope,
            index: 0,
        };
        self.store.insert(name.to_string(), symbol.clone());
        symbol
    }

    pub fn define_free(&mut self, original: Symbol) -> Symbol {
        self.free_symbols.push(original.clone());
        let symbol = Symbol {
            name: original.name,
            scope: SymbolScope::FreeScope,
            index: self.free_symbols.len() - 1,
        };
        self.store.insert(symbol.name.clone(), symbol.clone());
        symbol
    }

    pub fn resolve(&mut self, name: &str) -> Option<Symbol> {
        if let Some(symbol) = self.store.get(name) {
            Some(symbol.clone())
        } else if let Some(ref outer_rc) = self.outer {
            // Try to resolve in outer scope recursively first
            let resolved = {
                let mut outer_table = outer_rc.borrow_mut();
                outer_table.resolve(name)
            };
            
            if let Some(resolved_symbol) = resolved {
                if resolved_symbol.scope == SymbolScope::Global || resolved_symbol.scope == SymbolScope::Builtin {
                    return Some(resolved_symbol);
                } else {
                    // This is a free variable - convert it to FreeScope in our scope
                    let free = self.define_free(resolved_symbol);
                    return Some(free);
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn get_outer(&self) -> Option<Rc<RefCell<SymbolTable>>> {
        self.outer.clone()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define() {
        let mut expected = std::collections::HashMap::new();
        expected.insert("a".to_string(), Symbol {
            name: "a".to_string(),
            scope: SymbolScope::Global,
            index: 0,
        });
        expected.insert("b".to_string(), Symbol {
            name: "b".to_string(),
            scope: SymbolScope::Global,
            index: 1,
        });
        expected.insert("c".to_string(), Symbol {
            name: "c".to_string(),
            scope: SymbolScope::Local,
            index: 0,
        });
        expected.insert("d".to_string(), Symbol {
            name: "d".to_string(),
            scope: SymbolScope::Local,
            index: 1,
        });
        expected.insert("e".to_string(), Symbol {
            name: "e".to_string(),
            scope: SymbolScope::Local,
            index: 0,
        });
        expected.insert("f".to_string(), Symbol {
            name: "f".to_string(),
            scope: SymbolScope::Local,
            index: 1,
        });

        // global = NewSymbolTable()
        let mut global = SymbolTable::new();

        // a = global.Define("a")
        let a = global.define("a");
        assert_eq!(a, *expected.get("a").unwrap(), "expected a={:?}, got={:?}", expected.get("a").unwrap(), a);

        // b = global.Define("b")
        let b = global.define("b");
        assert_eq!(b, *expected.get("b").unwrap(), "expected b={:?}, got={:?}", expected.get("b").unwrap(), b);

        // firstLocal = NewEnclosedSymbolTable(global)
        let global_rc = Rc::new(RefCell::new(global));
        let mut first_local = SymbolTable::new_enclosed(Rc::clone(&global_rc));

        // c = firstLocal.Define("c")
        let c = first_local.define("c");
        assert_eq!(c, *expected.get("c").unwrap(), "expected c={:?}, got={:?}", expected.get("c").unwrap(), c);

        // d = firstLocal.Define("d")
        let d = first_local.define("d");
        assert_eq!(d, *expected.get("d").unwrap(), "expected d={:?}, got={:?}", expected.get("d").unwrap(), d);

        // secondLocal = NewEnclosedSymbolTable(firstLocal)
        let first_local_rc = Rc::new(RefCell::new(first_local));
        let mut second_local = SymbolTable::new_enclosed(Rc::clone(&first_local_rc));

        // e = secondLocal.Define("e")
        let e = second_local.define("e");
        assert_eq!(e, *expected.get("e").unwrap(), "expected e={:?}, got={:?}", expected.get("e").unwrap(), e);

        // f = secondLocal.Define("f")
        let f = second_local.define("f");
        assert_eq!(f, *expected.get("f").unwrap(), "expected f={:?}, got={:?}", expected.get("f").unwrap(), f);
    }


    #[test]
    fn test_resolve_global() {
        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");

        let expected = vec![
            Symbol {
                name: "a".to_string(),
                scope: SymbolScope::Global,
                index: 0,
            },
            Symbol {
                name: "b".to_string(),
                scope: SymbolScope::Global,
                index: 1,
            },
        ];

        for sym in expected {
            let result = global.resolve(&sym.name);
            if result.is_none() {
                panic!("name {} not resolvable", sym.name);
            }
            let result = result.unwrap();
            assert_eq!(
                result, sym,
                "expected {} to resolve to {:?}, got={:?}",
                sym.name, sym, result
            );
        }
    }

    #[test]
    fn test_resolve_local() {
        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");

        let global_rc = Rc::new(RefCell::new(global));
        let mut local = SymbolTable::new_enclosed(Rc::clone(&global_rc));
        local.define("c");
        local.define("d");

        let expected = vec![
            Symbol {
                name: "a".to_string(),
                scope: SymbolScope::Global,
                index: 0,
            },
            Symbol {
                name: "b".to_string(),
                scope: SymbolScope::Global,
                index: 1,
            },
            Symbol {
                name: "c".to_string(),
                scope: SymbolScope::Local,
                index: 0,
            },
            Symbol {
                name: "d".to_string(),
                scope: SymbolScope::Local,
                index: 1,
            },
        ];

        for sym in expected {
            let result = local.resolve(&sym.name);
            if result.is_none() {
                panic!("name {} not resolvable", sym.name);
            }
            let result = result.unwrap();
            assert_eq!(
                result, sym,
                "expected {} to resolve to {:?}, got={:?}",
                sym.name, sym, result
            );
        }
    }

    #[test]
    fn test_resolve_nested_local() {
        let mut global = SymbolTable::new();
        global.define("a");
        global.define("b");

        let global_rc = Rc::new(RefCell::new(global));
        let mut first_local = SymbolTable::new_enclosed(Rc::clone(&global_rc));
        first_local.define("c");
        first_local.define("d");

        let first_local_rc = Rc::new(RefCell::new(first_local));
        let mut second_local = SymbolTable::new_enclosed(Rc::clone(&first_local_rc));
        second_local.define("e");
        second_local.define("f");

        struct TestCase {
            table: Rc<RefCell<SymbolTable>>,
            expected_symbols: Vec<Symbol>,
        }

        let expected_first = vec![
            Symbol {
                name: "a".to_string(),
                scope: SymbolScope::Global,
                index: 0,
            },
            Symbol {
                name: "b".to_string(),
                scope: SymbolScope::Global,
                index: 1,
            },
            Symbol {
                name: "c".to_string(),
                scope: SymbolScope::Local,
                index: 0,
            },
            Symbol {
                name: "d".to_string(),
                scope: SymbolScope::Local,
                index: 1,
            },
        ];

        let expected_second = vec![
            Symbol {
                name: "a".to_string(),
                scope: SymbolScope::Global,
                index: 0,
            },
            Symbol {
                name: "b".to_string(),
                scope: SymbolScope::Global,
                index: 1,
            },
            Symbol {
                name: "e".to_string(),
                scope: SymbolScope::Local,
                index: 0,
            },
            Symbol {
                name: "f".to_string(),
                scope: SymbolScope::Local,
                index: 1,
            },
        ];

        let tests = vec![
            TestCase {
                table: Rc::clone(&first_local_rc),
                expected_symbols: expected_first,
            },
            TestCase {
                table: Rc::new(RefCell::new(second_local)),
                expected_symbols: expected_second,
            },
        ];

        for test in tests {
            for sym in test.expected_symbols {
                let result = test.table.borrow_mut().resolve(&sym.name);
                if result.is_none() {
                    panic!("name {} not resolvable", sym.name);
                }
                let result = result.unwrap();
                assert_eq!(
                    result, sym,
                    "expected {} to resolve to {:?}, got={:?}",
                    sym.name, sym, result
                );
            }
        }
    }

    #[test]
    fn test_define_resolve_builtins() {
        use super::*;

        // Create global and nested symbol tables
        let global = Rc::new(RefCell::new(SymbolTable::new()));
        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        let second_local = SymbolTable::new_enclosed(Rc::new(RefCell::new(first_local.clone())));

        // Expected builtin symbols
        let expected = vec![
            Symbol {
                name: "a".to_string(),
                scope: SymbolScope::Builtin,
                index: 0,
            },
            Symbol {
                name: "c".to_string(),
                scope: SymbolScope::Builtin,
                index: 1,
            },
            Symbol {
                name: "e".to_string(),
                scope: SymbolScope::Builtin,
                index: 2,
            },
            Symbol {
                name: "f".to_string(),
                scope: SymbolScope::Builtin,
                index: 3,
            },
        ];

        // Define builtins in the global symbol table
        for (i, sym) in expected.iter().enumerate() {
            global.borrow_mut().define_builtin(i, sym.name.clone());
        }

        // Test resolution in all tables
        let tables = vec![Rc::clone(&global), Rc::clone(&Rc::new(RefCell::new(first_local))), Rc::clone(&Rc::new(RefCell::new(second_local)))];
        for table in tables.iter() {
            for sym in &expected {
                let result = table.borrow_mut().resolve(&sym.name);
                if result.is_none() {
                    panic!("name {} not resolvable", sym.name);
                }
                let result = result.unwrap();
                assert_eq!(
                    result, *sym,
                    "expected {} to resolve to {:?}, got={:?}",
                    sym.name, sym, result
                );
            }
        }
    }

    #[test]
    fn test_resolve_free() {
        use crate::compiler::symbol_table::{SymbolTable, SymbolScope, Symbol};
        use std::rc::Rc;
        use std::cell::RefCell;

        let global = Rc::new(RefCell::new(SymbolTable::new()));
        global.borrow_mut().define("a");
        global.borrow_mut().define("b");

        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        let first_local_rc = Rc::new(RefCell::new(first_local));
        first_local_rc.borrow_mut().define("c");
        first_local_rc.borrow_mut().define("d");

        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local_rc));
        let second_local_rc = Rc::new(RefCell::new(second_local));
        second_local_rc.borrow_mut().define("e");
        second_local_rc.borrow_mut().define("f");

        let tests = vec![
            (
                Rc::clone(&first_local_rc),
                vec![
                    Symbol { name: "a".to_string(), scope: SymbolScope::Global, index: 0 },
                    Symbol { name: "b".to_string(), scope: SymbolScope::Global, index: 1 },
                    Symbol { name: "c".to_string(), scope: SymbolScope::Local, index: 0 },
                    Symbol { name: "d".to_string(), scope: SymbolScope::Local, index: 1 },
                ],
                vec![],
            ),
            (
                Rc::clone(&second_local_rc),
                vec![
                    Symbol { name: "a".to_string(), scope: SymbolScope::Global, index: 0 },
                    Symbol { name: "b".to_string(), scope: SymbolScope::Global, index: 1 },
                    Symbol { name: "c".to_string(), scope: SymbolScope::FreeScope, index: 0 },
                    Symbol { name: "d".to_string(), scope: SymbolScope::FreeScope, index: 1 },
                    Symbol { name: "e".to_string(), scope: SymbolScope::Local, index: 0 },
                    Symbol { name: "f".to_string(), scope: SymbolScope::Local, index: 1 },
                ],
                vec![
                    Symbol { name: "c".to_string(), scope: SymbolScope::Local, index: 0 },
                    Symbol { name: "d".to_string(), scope: SymbolScope::Local, index: 1 },
                ],
            ),
        ];

        for (table, expected_symbols, expected_free_symbols) in tests {
            for sym in &expected_symbols {
                let result = table.borrow_mut().resolve(&sym.name);
                assert!(result.is_some(), "name {} not resolvable", sym.name);
                let result = result.unwrap();
                assert_eq!(
                    result, *sym,
                    "expected {} to resolve to {:?}, got={:?}",
                    sym.name, sym, result
                );
            }

            let tbl = table.borrow();
            assert_eq!(
                tbl.free_symbols.len(),
                expected_free_symbols.len(),
                "wrong number of free symbols. got={}, want={}",
                tbl.free_symbols.len(),
                expected_free_symbols.len()
            );
            for (i, sym) in expected_free_symbols.iter().enumerate() {
                let result = &tbl.free_symbols[i];
                assert_eq!(result, sym, "wrong free symbol. got={:?}, want={:?}", result, sym);
            }
        }
    }

    #[test]
    fn test_resolve_unresolvable_free() {
        use crate::compiler::symbol_table::{SymbolTable, SymbolScope, Symbol};
        use std::rc::Rc;
        use std::cell::RefCell;

        let global = Rc::new(RefCell::new(SymbolTable::new()));
        global.borrow_mut().define("a");

        let first_local = SymbolTable::new_enclosed(Rc::clone(&global));
        let first_local_rc = Rc::new(RefCell::new(first_local));
        first_local_rc.borrow_mut().define("c");

        let second_local = SymbolTable::new_enclosed(Rc::clone(&first_local_rc));
        let second_local_rc = Rc::new(RefCell::new(second_local));
        second_local_rc.borrow_mut().define("e");
        second_local_rc.borrow_mut().define("f");

        let expected = vec![
            Symbol { name: "a".to_string(), scope: SymbolScope::Global, index: 0 },
            Symbol { name: "c".to_string(), scope: SymbolScope::FreeScope, index: 0 },
            Symbol { name: "e".to_string(), scope: SymbolScope::Local, index: 0 },
            Symbol { name: "f".to_string(), scope: SymbolScope::Local, index: 1 },
        ];

        for sym in &expected {
            let resolved = second_local_rc.borrow_mut().resolve(&sym.name);
            assert!(resolved.is_some(), "name {} not resolvable", sym.name);
            let resolved = resolved.unwrap();
            assert_eq!(
                resolved, *sym,
                "expected {} to resolve to {:?}, got={:?}",
                sym.name, sym, resolved
            );
        }

        let expected_unresolvable = vec![
            "b",
            "d",
        ];

        for name in expected_unresolvable {
            let resolved = second_local_rc.borrow_mut().resolve(name);
            assert!(resolved.is_none(), "name {} resolved, but was expected not to", name);
        }
    }

    #[test]
    fn test_define_and_resolve_function_name() {
        use crate::compiler::symbol_table::{SymbolTable, SymbolScope, Symbol};
        use std::rc::Rc;
        use std::cell::RefCell;

        let global = Rc::new(RefCell::new(SymbolTable::new()));
        global.borrow_mut().define_function_name("a");

        let expected = Symbol { name: "a".to_string(), scope: SymbolScope::FunctionScope, index: 0 };

        let result = global.borrow_mut().resolve(&expected.name);
        assert!(result.is_some(), "function name {} not resolvable", expected.name);

        let result = result.unwrap();
        assert_eq!(result, expected, "expected {} to resolve to {:?}, got={:?}", expected.name, expected, result);
    }

    #[test]
    fn test_shadowing_function_name() {
        use crate::compiler::symbol_table::{SymbolTable, SymbolScope, Symbol};
        use std::rc::Rc;
        use std::cell::RefCell;

        let global = Rc::new(RefCell::new(SymbolTable::new()));
        global.borrow_mut().define_function_name("a");
        global.borrow_mut().define("a");

        let expected = Symbol { name: "a".to_string(), scope: SymbolScope::Global, index: 0 };

        let result = global.borrow_mut().resolve(&expected.name);
        assert!(result.is_some(), "function name {} not resolvable", expected.name);

        let result = result.unwrap();
        assert_eq!(result, expected, "expected {} to resolve to {:?}, got={:?}", expected.name, expected, result);
    }
}

