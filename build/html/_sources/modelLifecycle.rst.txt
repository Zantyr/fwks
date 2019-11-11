Model lifecycle - training, evaluation and deployment
=====================================================

Building the model
------------------

Let us set up a basic speech enhancement model. The model will be trained on small testing corpus of recordings.
First, let's include all requirements from fwks library and prepare a dataset.

.. literalinclude:: examples/speech_enhance.py
   :lines: 1
   :linenos:

The training begins with the definition of the dataset. But before constructing the dataset, let's define the noise
which will be used during generation of the noisy samples. For our example, we will use simple Static class, which
contaminates the audio with white noise. The noise generator will be supplied to the dataset.

.. literalinclude:: examples/speech_enhance.py
   :lines: 1
   :linenos:

The dataset refereces a specific directory from which
files will be sourced. There are multiple details into the process of loading, however the default instantiation
should load all WAV files from a directory. The dataset itself does not include any files, therefore let's
source an example directory from fwks test suite.

.. literalinclude:: examples/speech_enhance.py
   :lines: 1
   :linenos:

The dataset object scans the directory, looking for files, however no data is produced. A dataset needs a proper
mapping and set of requirements in order to know how to preprocess the data. In most use cases that is supplied by
a Model, therefore let's create one.

.. literalinclude:: examples/speech_enhance.py
   :lines: 1
   :linenos:

The Model may consume the dataset using `.build` method. However, most common training regime is controlled
using Tasks. Let's create a DenoisingTask and add some metrics to it in order to be able to compare this particular
model with other designs.



Supplying a convenient path for caching the artifacts, launch the Task, via `.run()` method. The training should
begin and in the end an object with final results should be made and printed on the screen. An example output 
is like this:



Deployment
----------

Let's load another file and this time, let's load the previously saved modelS