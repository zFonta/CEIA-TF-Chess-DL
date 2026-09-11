"""Training-side code for the neural network block (WBS 4).

Only :mod:`chessdl.training.dataset` needs PyTorch. Splitting, caching and the
baselines are NumPy-only on purpose, so the parts that define the experiment can
be tested and reasoned about without a deep-learning stack installed.
"""
