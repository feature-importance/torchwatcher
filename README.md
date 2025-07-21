# TorchWatcher

`torchwatcher` uses `torch.fx` to trace and manipulate compute graphs of 
`PyTorch` models. Primarily, it has two purposes: modifying the forward and/or 
backward passes at specific points in a model; and extracting information 
from the forward/backward passes at specific points in order to do analysis 
of features, gradients, etc. These tasks are achieved by adding 
_interjections_ to the compute graph.

By using `torch.fx` we have the ability to work with models that already exist
without the need to modify their source code to add hooks. We also gain the 
ability to add interjections at points which would normally be difficult to 
access, such as calls to functions in `torch.nn.functional` which cannot be 
hooked with forward or backward hooks. 

## Low level APIs: Interjections

`torch.fx` creates a representation of the compute graph of a module through 
a process called tracing. The graph itself is represented by a collection of 
node objects which are linked together. Nodes in the graph represent 
operations such as calling a `nn.Module`s `forward` method, calling a 
function (e.g. something like `torch.relu` or `torch.nn.functional.relu`), 
calling an arbitrary method, etc. 

`torchwatcher`s interjection package allows you to:

- easily select node(s) in the graph where you want to place interjection 
- insert interjection either after each desired node, or, by wrapping the 
  desired node. The latter allows access to both the forward and backward 
  passes on the nodes, whilst the former only lets you access each node's 
  forward output.
- construct a new model that will run the interjection automatically on each 
  batch when you call it (and train it).

### Interjections  

Interjections are subclasses of one of:

- `torchwatcher.interjection.ForwardInterjection`
- `torchwatcher.interjection.WrappedForwardInterjection`
- `torchwatcher.interjection.WrappedForwardBackwardInterjection`

#### Forward Interjections 

`ForwardInterjection`s are inserted after a node in the compute graph. They 
are passed the selected node's output and are free to return a new value 
which will then be passed along to all the following nodes that originally 
consumed the selected node's output. In the case that the interjection does not 
return anything, the selected node's output is passed along unchanged.

Consider the following compute graph:

      ↓
    node_1
      ↓
    node_2
      ↓
    node_3
      ↓

If an instance of a subclass of `ForwardInterjection` (`my_interjection`) is 
inserted at`node_2`, the graph is transformed to  

           ↓
         node_1
           ↓
         node_2
           ↓
    my_interjection
           ↓
         node_3
           ↓

To use a `ForwardInterjection` you must create a subclass and override the 
`process` method:

    class MyForwardInterjection(ForwardInterjection):
        def process(self, name: str, module: nn.Module | None, inputs):
            print(name, inputs.shape)

    my_interjection = MyForwardInterjection()

The arguments to `process` are the respective node's name, the module (if 
the previous node was an `nn.Module`; `None` otherwise) and its outputs 
(presented as the inputs to the interjection). The node name is important 
because the same interjection instance might be inserted at multiple points 
in the graph. The name allows you to disambiguate which node is providing 
the input.  The type of the inputs to the method depends on what the 
selected node outputs; most typically it will be a single tensor, but it 
could potentially be a tuple of tensors, a dictionary or anything else.

#### Wrapped Interjections

Wrapped interjections work by inserting a wrapper around a node in the 
computational graph. The wrapper can access both the input to the wrapped 
node and its outputs. Taking the previous example, if a wrapped interjection 
is placed around `node_2`, the graph is transformed to

               ↓
             node_1
               ↓
    my_interjection(node_2)
               ↓
             node_3
               ↓

where internally `my_interjection` passes the output of `node_1` to `node_2` 
and returns the output of the `node_2` call. The interjection my alter the 
result of the `node_2` call but cannot modify the input (this could be 
achieved by placing an interjection on the previous node however).

We provide two base classes for wrapped interjections: the 
`WrappedForwardInterjection` class lets you capture the inputs and outputs, 
and the `WrappedForwardBackwardInterjection` additionally lets you capture the
computed gradients of the node in question. The latter is particularly nice 
as it lets you hook gradients for arbitrary parts of the graph, and not just 
calls to `nn.Module` instances - you can capture gradients for calls to 
`nn.functional` functions for example.

`WrappedForwardInterjection` require you implement a `process` method:

    class MyWrappedFwd(WrappedForwardInterjection):
        def process(self, name, module, inputs, outputs):
            print(name)

    my_interjection = MyWrappedFwd()

Using `WrappedForwardBackwardInterjection` involves implementing the 
`process_backward` method, and optionally, the `process` method if you also 
want to interject the forward pass

    class MyWrappedFwdBwd(WrappedForwardBackwardInterjection):
        def process(self, name, module, inputs, outputs):
            print(name)

        def process_backward(self, name, module, grad_input, grad_output):
            print(name)

    my_interjection = MyWrappedFwdBwd()

As with the forward interjection, the node name is passed to both `process` 
methods, together with the inputs and outputs (or input gradients and output 
gradients). If the interjection was wrapping an underlying `nn.Module` instance,
a reference to that instance is also passed along (this will be `None` if a 
different type of node in the compute graph was wrapped).

### Adding interjections to the graph

#### Inserting by matching a module

#### Inserting by name

#### Inserting using the node selector API

#### Tracing and setting custom leaf modules

## High-level APIs: Analysis and logging