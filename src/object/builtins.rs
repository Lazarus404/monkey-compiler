use crate::object::*;

pub const ARRAY_OBJ: &str = "ARRAY";

pub struct BuiltinFunction(pub fn(Vec<Object>) -> Object);

pub struct Builtin {
    pub function: BuiltinFunction,
}

pub struct BuiltinDef {
    pub name: &'static str,
    pub builtin: Builtin,
}

pub static BUILTINS: &[BuiltinDef] = &[
    BuiltinDef {
        name: "len",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                if args.len() != 1 {
                    return Object::Error(
                        format!(
                            "wrong number of arguments. got={}, want=1",
                            args.len()
                        ),
                    );
                }
                match &args[0] {
                    Object::Array(a) => Object::Int(a.len() as i64),
                    Object::String(s) => Object::Int(s.len() as i64),
                    other => Object::Error(format!(
                        "argument to `len` not supported, got {}",
                        other.r#type()
                    )),
                }
            }),
        },
    },
    BuiltinDef {
        name: "puts",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                for arg in &args {
                    println!("{}", arg.inspect());
                }
                Object::Null
            }),
        },
    },
    BuiltinDef {
        name: "first",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                if args.len() != 1 {
                    return Object::Error(
                        format!(
                            "wrong number of arguments. got={}, want=1",
                            args.len()
                        ),
                    );
                }
                if args[0].r#type() != ARRAY_OBJ {
                    return Object::Error(format!(
                        "argument to `first` must be ARRAY, got {}",
                        args[0].r#type()
                    ));
                }
                match &args[0] {
                    Object::Array(a) => {
                        if !a.is_empty() {
                            a[0].clone()
                        } else {
                            Object::Null
                        }
                    }
                    _ => Object::Null,
                }
            }),
        },
    },
    BuiltinDef {
        name: "last",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                if args.len() != 1 {
                    return Object::Error(
                        format!(
                            "wrong number of arguments. got={}, want=1",
                            args.len()
                        ),
                    );
                }
                if args[0].r#type() != ARRAY_OBJ {
                    return Object::Error(format!(
                        "argument to `last` must be ARRAY, got {}",
                        args[0].r#type()
                    ));
                }
                match &args[0] {
                    Object::Array(a) => {
                        if !a.is_empty() {
                            a[a.len() - 1].clone()
                        } else {
                            Object::Null
                        }
                    }
                    _ => Object::Null,
                }
            }),
        },
    },
    BuiltinDef {
        name: "rest",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                if args.len() != 1 {
                    return Object::Error(
                        format!(
                            "wrong number of arguments. got={}, want=1",
                            args.len()
                        ),
                    );
                }
                if args[0].r#type() != ARRAY_OBJ {
                    return Object::Error(format!(
                        "argument to `rest` must be ARRAY, got {}",
                        args[0].r#type()
                    ));
                }
                match &args[0] {
                    Object::Array(a) => {
                        if !a.is_empty() {
                            let new_elems = a[1..].to_vec();
                            Object::Array(new_elems)
                        } else {
                            Object::Null
                        }
                    }
                    _ => Object::Null,
                }
            }),
        },
    },
    BuiltinDef {
        name: "push",
        builtin: Builtin {
            function: BuiltinFunction(|args: Vec<Object>| {
                if args.len() != 2 {
                    return Object::Error(
                        format!(
                            "wrong number of arguments. got={}, want=2",
                            args.len()
                        ),
                    );
                }
                if args[0].r#type() != ARRAY_OBJ {
                    return Object::Error(format!(
                        "argument to `push` must be ARRAY, got {}",
                        args[0].r#type()
                    ));
                }
                match &args[0] {
                    Object::Array(a) => {
                        let mut new_elems = a.clone();
                        new_elems.push(args[1].clone());
                        Object::Array(new_elems)
                    }
                    _ => Object::Null,
                }
            }),
        },
    },
];

pub fn new_error(message: &str) -> Object {
    Object::Error(message.to_string())
}

pub fn get_builtin_by_name(name: &str) -> Option<&'static Builtin> {
    for def in BUILTINS {
        if def.name == name {
            return Some(&def.builtin);
        }
    }
    None
}
