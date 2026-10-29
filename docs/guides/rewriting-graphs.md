# Rewriting model graphs

In addition to interjections, `torchwatcher` also allows you to rewrite the compute graph. This is useful for things
like changing a model dynamically: for example, you could replace all instances of an `nn.Conv2d` with a different
module (for example a quantised version of the same module). Replacement is done through the use of a user-provided
callback function that is given a node in the graph and returns a new node.

`replace_module`: 1-to-1 replacement of a module in the graph, using the `torch.fx` tracing API to identify modules.

`replace_module_native`: 1-to-1 replacement of all modules in the model that match a given class.

