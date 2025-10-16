use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SymbolScope {
    Global,
    Local,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Symbol {
    pub name: String,
    pub scope: SymbolScope,
    pub index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SymbolTable {
    pub outer: Option<std::rc::Rc<std::cell::RefCell<SymbolTable>>>,
    store: std::collections::HashMap<String, Symbol>,
    pub num_definitions: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            outer: None,
            store: std::collections::HashMap::new(),
            num_definitions: 0,
        }
    }

    pub fn new_enclosed(outer: std::rc::Rc<std::cell::RefCell<SymbolTable>>) -> Self {
        SymbolTable {
            outer: Some(outer),
            store: std::collections::HashMap::new(),
            num_definitions: 0,
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

    pub fn resolve(&self, name: &str) -> Option<Symbol> {
        if let Some(symbol) = self.store.get(name) {
            Some(symbol.clone())
        } else if let Some(ref outer) = self.outer {
            // Recursively resolve from outer symbol table
            outer.borrow().resolve(name)
        } else {
            None
        }
    }

    pub fn get_outer(&self) -> Option<std::rc::Rc<std::cell::RefCell<SymbolTable>>> {
        self.outer.clone()
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_define() {
        use std::rc::Rc;
        use std::cell::RefCell;
        
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

        struct TestCase<'a> {
            table: &'a SymbolTable,
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

        let first_local_ref = second_local.get_outer().unwrap();
        let first_local_borrowed = first_local_ref.borrow();
        let tests = vec![
            TestCase {
                table: &first_local_borrowed, // first_local
                expected_symbols: expected_first,
            },
            TestCase {
                table: &second_local,
                expected_symbols: expected_second,
            },
        ];

        for test in tests {
            for sym in test.expected_symbols {
                let result = test.table.resolve(&sym.name);
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
}

