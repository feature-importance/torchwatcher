# torchwatcher

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
  passes on the nodes, whilst the former only lets you access each nodes 
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
        def process(self, name: str, inputs):
            print(name, inputs.shape)

    my_interjection = MyForwardInterjection()

The arguments to `process` are the respective node's name, and its outputs 
(presented as the inputs to the interjection). The node name is important 
because the same interjection instance might be inserted at multiple points 
in the graph. The name allows you to disambiguate which node is providing 
the input.  The type of the inputs to the method depends on what the 
selected node outputs; most typically it will be a single tensor, but it 
could potentially be a tuple of tensors, a dictionary or anything else.

#### Wrapped Interjections

### Adding interjections to the graph

#### Inserting by matching a module

#### Inserting by name

#### Inserting using the node selector API


## High-level APIs: Analysis and logging