.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_advanced_dynamic_quantization_tutorial.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_advanced_dynamic_quantization_tutorial.py:


(beta) Dynamic Quantization on an LSTM Word Language Model
==================================================================

**Author**: `James Reed <https://github.com/jamesr66a>`_

**Edited by**: `Seth Weidman <https://github.com/SethHWeidman/>`_

Introduction
------------

Quantization involves converting the weights and activations of your model from float
to int, which can result in smaller model size and faster inference with only a small
hit to accuracy.

In this tutorial, we'll apply the easiest form of quantization -
`dynamic quantization <https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic>`_ -
to an LSTM-based next word-prediction model, closely following the
`word language model <https://github.com/pytorch/examples/tree/master/word_language_model>`_
from the PyTorch examples.

.. code-block:: default


    # imports
    import os
    from io import open
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F







1. Define the model
-------------------

Here we define the LSTM model architecture, following the
`model <https://github.com/pytorch/examples/blob/master/word_language_model/model.py>`_
from the word language model example.


.. code-block:: default


    class LSTMModel(nn.Module):
        """Container module with an encoder, a recurrent module, and a decoder."""

        def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
            super(LSTMModel, self).__init__()
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(ntoken, ninp)
            self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)

            self.init_weights()

            self.nhid = nhid
            self.nlayers = nlayers

        def init_weights(self):
            initrange = 0.1
            self.encoder.weight.data.uniform_(-initrange, initrange)
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-initrange, initrange)

        def forward(self, input, hidden):
            emb = self.drop(self.encoder(input))
            output, hidden = self.rnn(emb, hidden)
            output = self.drop(output)
            decoded = self.decoder(output)
            return decoded, hidden

        def init_hidden(self, bsz):
            weight = next(self.parameters())
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))







2. Load in the text data
------------------------

Next, we load the
`Wikitext-2 dataset <https://www.google.com/search?q=wikitext+2+data>`_ into a `Corpus`,
again following the
`preprocessing <https://github.com/pytorch/examples/blob/master/word_language_model/data.py>`_
from the word language model example.


.. code-block:: default


    class Dictionary(object):
        def __init__(self):
            self.word2idx = {}
            self.idx2word = []

        def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)


    class Corpus(object):
        def __init__(self, path):
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

        def tokenize(self, path):
            """Tokenizes a text file."""
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    words = line.split() + ['<eos>']
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)

            return ids

    model_data_filepath = 'data/'

    corpus = Corpus(model_data_filepath + 'wikitext-2')







3. Load the pre-trained model
-----------------------------

This is a tutorial on dynamic quantization, a quantization technique
that is applied after a model has been trained. Therefore, we'll simply load some
pre-trained weights into this model architecture; these weights were obtained
by training for five epochs using the default settings in the word language model
example.


.. code-block:: default


    ntokens = len(corpus.dictionary)

    model = LSTMModel(
        ntoken = ntokens,
        ninp = 512,
        nhid = 256,
        nlayers = 5,
    )

    model.load_state_dict(
        torch.load(
            model_data_filepath + 'word_language_model_quantize.pth',
            map_location=torch.device('cpu')
            )
        )

    model.eval()
    print(model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    LSTMModel(
      (drop): Dropout(p=0.5, inplace=False)
      (encoder): Embedding(33278, 512)
      (rnn): LSTM(512, 256, num_layers=5, dropout=0.5)
      (decoder): Linear(in_features=256, out_features=33278, bias=True)
    )


Now let's generate some text to ensure that the pre-trained model is working
properly - similarly to before, we follow
`here <https://github.com/pytorch/examples/blob/master/word_language_model/generate.py>`_


.. code-block:: default


    input_ = torch.randint(ntokens, (1, 1), dtype=torch.long)
    hidden = model.init_hidden(1)
    temperature = 1.0
    num_words = 1000

    with open(model_data_filepath + 'out.txt', 'w') as outf:
        with torch.no_grad():  # no tracking history
            for i in range(num_words):
                output, hidden = model(input_, hidden)
                word_weights = output.squeeze().div(temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input_.fill_(word_idx)

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(str(word.encode('utf-8')) + ('\n' if i % 20 == 19 else ' '))

                if i % 100 == 0:
                    print('| Generated {}/{} words'.format(i, 1000))

    with open(model_data_filepath + 'out.txt', 'r') as outf:
        all_output = outf.read()
        print(all_output)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    | Generated 0/1000 words
    | Generated 100/1000 words
    | Generated 200/1000 words
    | Generated 300/1000 words
    | Generated 400/1000 words
    | Generated 500/1000 words
    | Generated 600/1000 words
    | Generated 700/1000 words
    | Generated 800/1000 words
    | Generated 900/1000 words
    b'martial' b'diversity' b'from' b'the' b'dwarf' b'Panther' b'Lakes' b'War' b'and' b'the' b'character' b'in' b'2009' b'.' b'<eos>' b'<eos>' b'=' b'=' b'Themes' b'='
    b'=' b'<eos>' b'<eos>' b'The' b'majority' b'of' b'1' b'@.@' b'7' b'@.@' b'5' b'million' b'ago' b'(' b'51' b'@.@' b'8' b'in' b')' b'irresistible'
    b'corridor' b'was' b'introduced' b'of' b'important' b'events' b',' b'designating' b'it' b'a' b'model' b'of' b'Ohio' b'design' b'in' b'rats' b',' b'and' b'South' b'Angeles'
    b'.' b'In' b'1939' b',' b'the' b'number' b'of' b'the' b'steps' b'from' b'Christ' b'also' b'states' b'a' b'planet' b'to' b'date' b'that' b'proceeds' b','
    b'and' b'has' b'moved' b'to' b'Rashid' b',' b'creating' b'it' b'Labour' b'<unk>' b'of' b'Jacksons' b'as' b'render' b'on' b'that' b'of' b'Dublin' b'available' b'.'
    b'telling' b'fuel' b'textile' b'laws' b',' b'they' b'also' b'have' b'determined' b'immediate' b'evidence' b'of' b'agricultural' b'starling' b',' b'support' b'threat' b'dealing' b'.' b'<eos>'
    b'The' b'clutch' b'and' b'neo' b'@-@' b'language' b'agricultural' b'Ukrainian' b'<unk>' b',' b'food' b'south' b',' b'is' b'trapped' b'.' b'No.' b'3' b'on' b'the'
    b'ensuing' b'spot' b'.' b'This' b'is' b'driven' b'to' b'average' b'different' b'hi' b'species' b';' b'this' b'elaborate' b'descriptions' b'of' b'connections' b'became' b'also' b'native'
    b'and' b'therefore' b'to' b'be' b'placed' b'to' b'stimulate' b'culture' b',' b'which' b'do' b'not' b'become' b'strong' b'as' b'they' b'show' b',' b'as' b'may'
    b'be' b'relocated' b'as' b'"' b'the' b'ventral' b'being' b'<unk>' b'suit' b'"' b'.' b'The' b'stem' b'can' b'be' b'R' b'Age' b',' b'as' b'they'
    b'are' b'distinguished' b',' b'release' b',' b'and' b'<unk>' b'.' b'The' b'congregation' b'@-@' b'shaped' b',' b'designated' b'by' b'M.' b'Johnson' b',' b'contains' b'prey'
    b'attributed' b'south' b'.' b'A' b'document' b'disappeared' b'between' b'stone' b'and' b'Chinese' b'Island' b'+' b'Colombia' b'where' b'The' b'kakapo' b'mimicry' b'can' b'be' b'distinguished'
    b'with' b'1638' b'.' b'Thereafter' b',' b'M.' b'coaster' b'has' b'a' b'rare' b'field' b'to' b'God' b',' b'preceded' b'as' b'it' b'chairman' b'I.' b'Revenge'
    b'and' b'Nefer' b'Carl' b'Jones' b'.' b'As' b'there' b'had' b'also' b'disorganised' b'associations' b',' b'this' b'bird' b'will' b'be' b'distinguished' b'by' b'three' b'wine'
    b'starlings' b',' b'including' b'them' b'to' b'Scotland' b',' b'hence' b'a' b'population' b'for' b'$' b'50' b'million' b'in' b'damage' b',' b'and' b'in' b'particular'
    b'Yo' b'Bryant' b'may' b'be' b'significantly' b'followed' b'and' b'a' b'refrain' b'.' b'More' b'than' b'only' b'they' b'making' b'to' b'year' b'at' b'times' b'.'
    b'It' b'may' b'be' b'entirely' b'recommended' b'between' b'other' b'birds' b'and' b'second' b'12th' b'or' b'both' b'centuries' b'.' b'The' b'image' b'with' b'generally' b':'
    b'A' b'row' b'to' b'introduce' b'our' b',' b'and' b'indicate' b'they' b'could' b'have' b'enjoyed' b'they' b'can' b'be' b'.' b'<eos>' b'A' b'more' b'different'
    b'scheme' b'used' b'by' b'Papal' b'from' b'<unk>' b'distance' b'to' b'the' b'ground' b')' b'is' b'the' b'teens' b'Monastery' b'exactly' b'by' b'its' b'32' b'@-@'
    b'European' b'survivals' b'.' b'Each' b'female' b'is' b'typically' b'visible' b'in' b'the' b'light' b'inscription' b'of' b'foods' b',' b'a' b'margin' b'of' b'individually' b','
    b'sunlight' b';' b'towers' b'and' b'driving' b'the' b'chicks' b'talk' b'within' b'Asomtavruli' b',' b'its' b'supposed' b'they' b'might' b'be' b'found' b'by' b'up' b'least'
    b'far' b'on' b'humans' b'.' b'There' b'was' b'also' b'plans' b'females' b'project' b'harvest' b'is' b'known' b'any' b'measure' b'.' b'However' b',' b'they' b'may'
    b'be' b'immediately' b'effective' b'or' b'introduced' b'to' b'bring' b'the' b'ailing' b'supposed' b'Wu' b'Inquiry' b'.' b'Unlike' b'a' b'nest' b'possession' b'for' b'birds' b','
    b'he' b'usually' b'chloride' b'to' b'be' b'short' b'well' b',' b'typically' b'<unk>' b'thin' b',' b'most' b'other' b',' b'Zabibe' b'or' b'beg' b',' b'distinct'
    b'towards' b'their' b'stones' b',' b'and' b'its' b'worst' b'armament' b',' b'dissolve' b'rescues' b',' b'mature' b'instruction' b'and' b'rich' b'motifs' b'.' b'"' b'3rd'
    b'He' b'cannot' b'be' b'"' b'while' b'they' b'represent' b'that' b'it' b'be' b'stronger' b'to' b'cook' b'crescent' b'(' b'1997' b')' b'when' b'common' b'as'
    b'their' b'best' b'part' b'@-@' b'eggs' b'so' b'its' b'beak' b'only' b'when' b'they' b'were' b'<unk>' b',' b'including' b'some' b'farther' b'(' b'or' b'of'
    b'those' b'people' b')' b',' b'as' b'soon' b'in' b'.' b'Moors' b'and' b'<unk>' b'citizens' b'has' b'rarely' b'stole' b'each' b'of' b'the' b'species' b'to'
    b'be' b'poor' b'.' b'Later' b'activities' b'are' b'limited' b'to' b'their' b'pursuers' b'throughout' b'its' b'raising' b'lines' b'<unk>' b',' b'they' b'can' b'potentially' b'share'
    b'her' b'right' b'.' b'One' b'vapor' b'means' b'them' b'iconography' b',' b'sports' b'their' b'<unk>' b'bag' b';' b'and' b'Raghavanka' b'blunt' b'that' b'navigation' b','
    b'branched' b',' b'and' b'currants' b',' b'form' b'the' b'fly' b'star' b'room' b'.' b'Project' b'chemotherapy' b'and' b'artificial' b'descriptions' b'of' b'body' b'favorites' b'are'
    b'composed' b'.' b'For' b'many' b'variable' b'times' b',' b'intermediate' b',' b'their' b'snakes' b'is' b'here' b'.' b'When' b'it' b'evolved' b'to' b'culinary' b'feathers'
    b',' b'they' b'may' b'be' b'made' b'.' b'Mhalsa' b'has' b'come' b'to' b'be' b'distributed' b'by' b'contraception' b'.' b'Some' b'toxic' b'spends' b'abundance' b'they'
    b'pass' b'in' b'the' b'common' b'heritage' b'of' b'gable' b'.' b'As' b'they' b'never' b'lie' b'two' b',' b'when' b'they' b'may' b'be' b'printed' b'drama'
    b',' b'he' b'hold' b'decibels' b',' b'so' b'as' b'some' b'earlier' b'authorities' b'indicates' b'in' b'their' b'water' b'.' b'A' b'computer' b'associated' b'during' b'a'
    b'cargo' b'by' b'double' b'sporadically' b'from' b'ticket' b'exposed' b'in' b'most' b'cases' b',' b'but' b'when' b'they' b'involve' b'respect' b'.' b'<eos>' b'Ceres' b"'s"
    b'wedding' b'area' b'is' b'integrated' b'inside' b'Islamic' b'wheat' b',' b'which' b'was' b'prominent' b'hunted' b'(' b'761' b"'s" b'tomb' b')' b',' b'and' b'was'
    b'fully' b'referred' b'to' b'as' b'"' b'about' b'$' b'80' b'undeveloped' b'"' b'.' b'In' b'both' b'quality' b',' b'large' b'hypothesis' b'was' b'published' b'by'
    b'native' b'rows' b'(' b'although' b'habit' b'have' b'their' b'other' b'of' b'Ceres' b'at' b'"' b'waste' b'"' b'officers' b')' b'.' b'Males' b'needs' b'to'
    b'their' b'sunken' b'body' b',' b'so' b'lacking' b'spring' b'chemotherapy' b'and' b'coastline' b'they' b'will' b'rarely' b'Goldwyn' b',' b'particularly' b'of' b'which' b'each' b'has'
    b'a' b'active' b'star' b'state' b'which' b'takes' b'15' b'%' b'of' b'her' b'person' b'.' b'In' b'the' b'eastern' b'nozzle' b',' b'Asia' b'dated' b'no'
    b'names' b'at' b'least' b',' b'and' b'it' b'is' b'one' b'of' b'Supernatural' b'reported' b'that' b'some' b'of' b'Ceres' b'or' b'diseases' b'are' b'tune' b'.'
    b'<eos>' b'One' b'present' b'birds' b'that' b'occur' b'in' b'Australia' b',' b'bright' b'starlings' b'were' b'repeatedly' b'vague' b'.' b'Because' b'of' b'its' b'call' b','
    b'he' b'may' b'also' b'be' b'capable' b'of' b'those' b'evoke' b',' b'even' b'very' b'non' b'@-@' b'year' b'with' b'ages' b'.' b'Since' b'attempting' b'to'
    b'leave' b'the' b'nest' b'of' b'<unk>' b',' b'tops' b'may' b'have' b'on' b'2' b'October' b',' b'through' b'them' b'to' b'deliver' b'them' b'.' b'An'
    b'last' b'engraving' b',' b'suplex' b'impact' b',' b'rapid' b'poisoning' b',' b'planted' b'low' b'methods' b',' b'<unk>' b'Anonymus' b',' b'mobbing' b',' b'Norway' b','
    b'shown' b',' b'<unk>' b',' b'griffin' b',' b'and' b'small' b'approaches' b'themselves' b'.' b'<eos>' b'Shruti' b'and' b'plants' b'indicate' b'that' b'fruits' b'carry' b'around'
    b'all' b'.' b'Operationally' b',' b'that' b'rays' b'are' b'rebuilt' b'.' b'In' b'250' b'parallax' b'ecological' b'starlings' b',' b'eggs' b'out' b'or' b'two' b'divorced'
    b'branches' b'are' b'known' b'to' b'have' b'been' b'seen' b'at' b'a' b'long' b'moment' b'of' b'kakapo' b'(' b'lay' b')' b'.' b'<eos>' b'In' b'Islay'
    b',' b'they' b'were' b'wiped' b'to' b'favor' b'of' b'Jupiter' b"'" b'technique' b'.' b'Because' b'minerals' b'sites' b'are' b'uncommon' b'.' b'Formation' b'wig' b'include'


It's no GPT-2, but it looks like the model has started to learn the structure of
language!

We're almost ready to demonstrate dynamic quantization. We just need to define a few more
helper functions:


.. code-block:: default


    bptt = 25
    criterion = nn.CrossEntropyLoss()
    eval_batch_size = 1

    # create test data set
    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        return data.view(bsz, -1).t().contiguous()

    test_data = batchify(corpus.test, eval_batch_size)

    # Evaluation functions
    def get_batch(source, i):
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target

    def repackage_hidden(h):
      """Wraps hidden states in new Tensors, to detach them from their history."""

      if isinstance(h, torch.Tensor):
          return h.detach()
      else:
          return tuple(repackage_hidden(v) for v in h)

    def evaluate(model_, data_source):
        # Turn on evaluation mode which disables dropout.
        model_.eval()
        total_loss = 0.
        hidden = model_.init_hidden(eval_batch_size)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, bptt):
                data, targets = get_batch(data_source, i)
                output, hidden = model_(data, hidden)
                hidden = repackage_hidden(hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)







4. Test dynamic quantization
----------------------------

Finally, we can call ``torch.quantization.quantize_dynamic`` on the model!
Specifically,

- We specify that we want the ``nn.LSTM`` and ``nn.Linear`` modules in our
  model to be quantized
- We specify that we want weights to be converted to ``int8`` values


.. code-block:: default


    import torch.quantization

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
    )
    print(quantized_model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    LSTMModel(
      (drop): Dropout(p=0.5, inplace=False)
      (encoder): Embedding(33278, 512)
      (rnn): DynamicQuantizedLSTM(512, 256, num_layers=5, dropout=0.5)
      (decoder): DynamicQuantizedLinear(in_features=256, out_features=33278, dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    )


The model looks the same; how has this benefited us? First, we see a
significant reduction in model size:


.. code-block:: default


    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

    print_size_of_model(model)
    print_size_of_model(quantized_model)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Size (MB): 113.94579
    Size (MB): 79.739984


Second, we see faster inference time, with no difference in evaluation loss:

Note: we set the number of threads to one for single threaded comparison, since quantized
models run single threaded.


.. code-block:: default


    torch.set_num_threads(1)

    def time_model_evaluation(model, test_data):
        s = time.time()
        loss = evaluate(model, test_data)
        elapsed = time.time() - s
        print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

    time_model_evaluation(model, test_data)
    time_model_evaluation(quantized_model, test_data)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    loss: 5.167
    elapsed time (seconds): 225.9
    loss: 5.168
    elapsed time (seconds): 145.7


Running this locally on a MacBook Pro, without quantization, inference takes about 200 seconds,
and with quantization it takes just about 100 seconds.

Conclusion
----------

Dynamic quantization can be an easy way to reduce model size while only
having a limited effect on accuracy.

Thanks for reading! As always, we welcome any feedback, so please create an issue
`here <https://github.com/pytorch/pytorch/issues>`_ if you have any.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 7 minutes  18.400 seconds)


.. _sphx_glr_download_advanced_dynamic_quantization_tutorial.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: dynamic_quantization_tutorial.py <dynamic_quantization_tutorial.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: dynamic_quantization_tutorial.ipynb <dynamic_quantization_tutorial.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
