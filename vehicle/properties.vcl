--------------------------------------------------------------------------------
-- Inputs

-- We first define a new name for the type of inputs of the network.
-- In particular, it takes inputs of the form of a vector of m rational numbers
-- (the size depends).

m = 5

inputSize : Nat
inputSize = 2 + 4 * m

minIAT  = 0.0000001 -- seconds are: IATs * 50000000000
maxIAT  = 0.000005 -- seconds are: IATs * 50000000000
minSIZE = 40 / 1000
maxSIZE = 52 / 1000

type InputVector = Tensor Rat [inputSize]

-- Next we add meaningful names for the indices.

timeElapsed       =  0
protocol          =  1

pktDirection1     =  2 -- 2 + 0 * m + 0
pktDirection2     =  3 -- 2 + 0 * m + 1
pktDirection3     =  4 -- 2 + 0 * m + 2
pktDirection4     =  5 -- 2 + 0 * m + 3
pktDirection5     =  6 -- 2 + 0 * m + 4

pktFlags1         =  7 -- 2 + 1 * m + 0
pktFlags2         =  8 -- 2 + 1 * m + 1
pktFlags3         =  9 -- 2 + 1 * m + 2
pktFlags4         = 10 -- 2 + 1 * m + 3
pktFlags5         = 11 -- 2 + 1 * m + 4

pktIATs1          = 12 -- 2 + 2 * m + 0
pktIATs2          = 13 -- 2 + 2 * m + 1
pktIATs3          = 14 -- 2 + 2 * m + 2
pktIATs4          = 15 -- 2 + 2 * m + 3
pktIATs5          = 16 -- 2 + 2 * m + 4

pktSize1          = 17 -- 2 + 3 * m + 0
pktSize2          = 18 -- 2 + 3 * m + 1
pktSize3          = 19 -- 2 + 3 * m + 2
pktSize4          = 20 -- 2 + 3 * m + 3
pktSize5          = 21 -- 2 + 3 * m + 4

--------------------------------------------------------------------------------
-- Outputs

-- Outputs are a vector of 2 rationals. Representing the POS and NEG classes.

type OutputVector = Vector Rat 2
type Label = Index 2

-- Again we define meaningful names for the indices into output vectors.

pos = 0
neg = 1

--------------------------------------------------------------------------------
-- The network

-- Next we use the `network` annotation to declare the name and the type of the
-- neural network we are verifying. The implementation is passed to the compiler
-- via a reference to the ONNX file at compile time.

@network
classifier : InputVector -> OutputVector

-- The classifier advises that input vector `x` has label `i` if the score
-- for label `i` is greater than the score of any other label `j`.
advises : InputVector -> Label -> Bool
advises x i = forall j . j != i => classifier x ! i > classifier x ! j

--------------------------------------------------------------------------------
-- Functions

validInput : InputVector -> Bool
validInput x = forall i . 0.0 <= x ! i <= 1.0

bounded : InputVector -> InputVector -> Bool
bounded input x =
  x ! 0 == input ! 0 and
  x ! 1 == input ! 1 and
  x ! 2 == input ! 2 and
  x ! 3 == input ! 3 and
  x ! 4 == input ! 4 and
  x ! 5 == input ! 5 and
  x ! 6 == input ! 6 and
  x ! 7 == input ! 7 and
  x ! 8 == input ! 8 and
  x ! 9 == input ! 9 and
  x ! 10 == input ! 10 and
  x ! 11 == input ! 11 and
  x ! 12 == input ! 12 and
  x ! 13 == input ! 13 and
  minIAT <= x ! pktIATs3 <= maxIAT and
  x ! 15 == input ! 15 and
  x ! 16 == input ! 16 and
  x ! 17 == input ! 17 and
  x ! 18 == input ! 18 and
  minSIZE <= x ! pktSize3 <= maxSIZE and
  x ! 20 == input ! 20 and
  x ! 21 == input ! 21

-- Robust single input
robust : InputVector -> Bool
robust input = forall x . bounded input x and validInput x =>
    advises x pos

--------------------------------------------------------------------------------
@parameter(infer=True)
n : Nat

@dataset
trainingInputs : Vector InputVector n

--------------------------------------------------------------------------------
-----------------------------------PROPERTY-------------------------------------
--------------------------------------------------------------------------------
-- Property: for each input in the training dataset, the robust property holds.

@property
property : Vector Bool n
property = foreach i . robust (trainingInputs ! i)
--------------------------------------------------------------------------------