use crate::object::{Object, builtins::get_builtin_by_name};
use std::collections::HashMap;

/// Returns a HashMap of the built-in function names to their Object::Builtin instances.
pub fn new_builtins() -> HashMap<String, Object> {
    let mut builtins = HashMap::new();
    let names = ["len", "puts", "first", "last", "rest", "push"];
    let param_counts = [1, -1, 1, 1, 1, 2]; // -1 for puts means variable arguments
    
    for (i, name) in names.iter().enumerate() {
        if let Some(builtin) = get_builtin_by_name(name) {
            builtins.insert((*name).to_string(), Object::Builtin(param_counts[i], builtin.function.0));
        }
    }
    builtins
}
