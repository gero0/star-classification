??
? ? 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
DenseBincount
input"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
.
Identity

input"T
output"T"	
Ttype
$

LogicalAnd
x

y

z
?
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Min

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ֻ
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1622317*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1622236*
value_dtype0	
q
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1622446*
value_dtype0	
?
MutableHashTable_1MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1622365*
value_dtype0	
\
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean
U
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
: *
dtype0
d
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance
]
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
`
mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_1
Y
mean_1/Read/ReadVariableOpReadVariableOpmean_1*
_output_shapes
: *
dtype0
h

variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_1
a
variance_1/Read/ReadVariableOpReadVariableOp
variance_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0	
`
mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_2
Y
mean_2/Read/ReadVariableOpReadVariableOpmean_2*
_output_shapes
: *
dtype0
h

variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_2
a
variance_2/Read/ReadVariableOpReadVariableOp
variance_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0	
`
mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namemean_3
Y
mean_3/Read/ReadVariableOpReadVariableOpmean_3*
_output_shapes
: *
dtype0
h

variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
variance_3
a
variance_3/Read/ReadVariableOpReadVariableOp
variance_3*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0	
v
Hidden/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameHidden/kernel
o
!Hidden/kernel/Read/ReadVariableOpReadVariableOpHidden/kernel*
_output_shapes

:*
dtype0
n
Hidden/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameHidden/bias
g
Hidden/bias/Read/ReadVariableOpReadVariableOpHidden/bias*
_output_shapes
:*
dtype0
v
Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameOutput/kernel
o
!Output/kernel/Read/ReadVariableOpReadVariableOpOutput/kernel*
_output_shapes

:*
dtype0
n
Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameOutput/bias
g
Output/bias/Read/ReadVariableOpReadVariableOpOutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
?
Adam/Hidden/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Hidden/kernel/m
}
(Adam/Hidden/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Hidden/kernel/m*
_output_shapes

:*
dtype0
|
Adam/Hidden/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Hidden/bias/m
u
&Adam/Hidden/bias/m/Read/ReadVariableOpReadVariableOpAdam/Hidden/bias/m*
_output_shapes
:*
dtype0
?
Adam/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Output/kernel/m
}
(Adam/Output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/m*
_output_shapes

:*
dtype0
|
Adam/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/m
u
&Adam/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output/bias/m*
_output_shapes
:*
dtype0
?
Adam/Hidden/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Hidden/kernel/v
}
(Adam/Hidden/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Hidden/kernel/v*
_output_shapes

:*
dtype0
|
Adam/Hidden/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Hidden/bias/v
u
&Adam/Hidden/bias/v/Read/ReadVariableOpReadVariableOpAdam/Hidden/bias/v*
_output_shapes
:*
dtype0
?
Adam/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/Output/kernel/v
}
(Adam/Output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output/kernel/v*
_output_shapes

:*
dtype0
|
Adam/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/Output/bias/v
u
&Adam/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R 
\
Const_2Const*
_output_shapes

:*
dtype0*
valueB*?F
\
Const_3Const*
_output_shapes

:*
dtype0*
valueB*??L
\
Const_4Const*
_output_shapes

:*
dtype0*
valueB*R??G
\
Const_5Const*
_output_shapes

:*
dtype0*
valueB*˪?P
\
Const_6Const*
_output_shapes

:*
dtype0*
valueB*?בC
\
Const_7Const*
_output_shapes

:*
dtype0*
valueB*??H
\
Const_8Const*
_output_shapes

:*
dtype0*
valueB*???@
\
Const_9Const*
_output_shapes

:*
dtype0*
valueB*:??B
J
Const_10Const*
_output_shapes
: *
dtype0	*
value	B	 R 
J
Const_11Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_12Const*
_output_shapes
:*
dtype0*?
value?B?BREDBBLUEB
BLUE-WHITEBWHITEBYELLOW-WHITEBYELLOWISH WHITEBWHITISHB	YELLOWISHBWHITE-YELLOWBPALE YELLOW ORANGEBORANGEBBLUE-WHITE 
?
Const_13Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`                                                        	       
                     
d
Const_14Const*
_output_shapes
:*
dtype0*(
valueBBMBOBBBABFBKBG
?
Const_15Const*
_output_shapes
:*
dtype0	*M
valueDBB	"8                                                 
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_12Const_13*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_1648764
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_1648769
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_14Const_15*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_1648777
?
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *%
f R
__inference_<lambda>_1648782
h
NoOpNoOp^PartitionedCall^PartitionedCall_1^StatefulPartitionedCall^StatefulPartitionedCall_1
?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable*
Tkeys0*
Tvalues0	*#
_class
loc:@MutableHashTable*
_output_shapes

::
?
AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2LookupTableExportV2MutableHashTable_1*
Tkeys0*
Tvalues0	*%
_class
loc:@MutableHashTable_1*
_output_shapes

::
?G
Const_16Const"/device:CPU:0*
_output_shapes
: *
dtype0*?F
value?FB?F B?F
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
* 
* 
* 
* 
L
lookup_table
token_counts
	keras_api
_adapt_function*
L
lookup_table
 token_counts
!	keras_api
"_adapt_function*
?
#
_keep_axis
$_reduce_axis
%_reduce_axis_mask
&_broadcast_shape
'mean
'
adapt_mean
(variance
(adapt_variance
	)count
*	keras_api
+_adapt_function*
?
,
_keep_axis
-_reduce_axis
._reduce_axis_mask
/_broadcast_shape
0mean
0
adapt_mean
1variance
1adapt_variance
	2count
3	keras_api
4_adapt_function*
?
5
_keep_axis
6_reduce_axis
7_reduce_axis_mask
8_broadcast_shape
9mean
9
adapt_mean
:variance
:adapt_variance
	;count
<	keras_api
=_adapt_function*
?
>
_keep_axis
?_reduce_axis
@_reduce_axis_mask
A_broadcast_shape
Bmean
B
adapt_mean
Cvariance
Cadapt_variance
	Dcount
E	keras_api
F_adapt_function*
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses* 
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses* 
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
?

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses*
?
iiter

jbeta_1

kbeta_2
	ldecay
mlearning_rateYm?Zm?am?bm?Yv?Zv?av?bv?*
|
'2
(3
)4
05
16
27
98
:9
;10
B11
C12
D13
Y14
Z15
a16
b17*
 
Y0
Z1
a2
b3*
* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

sserving_default* 
R
t_initializer
u_create_resource
v_initialize
w_destroy_resource* 
?
x_create_resource
y_initialize
z_destroy_resource<
table3layer_with_weights-0/token_counts/.ATTRIBUTES/table*
* 
* 
R
{_initializer
|_create_resource
}_initialize
~_destroy_resource* 
?
_create_resource
?_initialize
?_destroy_resource<
table3layer_with_weights-1/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
* 
RL
VARIABLE_VALUEmean4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEvariance8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEcount5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
TN
VARIABLE_VALUEmean_14layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUE
variance_18layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_15layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
TN
VARIABLE_VALUEmean_24layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUE
variance_28layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_25layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
TN
VARIABLE_VALUEmean_34layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUE
variance_38layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEcount_35layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEHidden/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEHidden/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

a0
b1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
\
'2
(3
)4
05
16
27
98
:9
;10
B11
C12
D13*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

?total

?count
?	variables
?	keras_api*
M

?total

?count
?
_fn_kwargs
?	variables
?	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_54keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

?0
?1*

?	variables*
?z
VARIABLE_VALUEAdam/Hidden/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Hidden/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Hidden/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Hidden/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
!serving_default_AbsoluteMagnitudePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_LuminosityPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_RadiusPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_SpectralClassPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_StarColorPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_TemperaturePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCall_2StatefulPartitionedCall!serving_default_AbsoluteMagnitudeserving_default_Luminosityserving_default_Radiusserving_default_SpectralClassserving_default_StarColorserving_default_Temperaturehash_table_1Const
hash_tableConst_1Const_2Const_3Const_4Const_5Const_6Const_7Const_8Const_9Hidden/kernelHidden/biasOutput/kernelOutput/bias*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1648281
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2CMutableHashTable_1_lookup_table_export_values/LookupTableExportV2:1mean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOpmean_1/Read/ReadVariableOpvariance_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpmean_2/Read/ReadVariableOpvariance_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpmean_3/Read/ReadVariableOpvariance_3/Read/ReadVariableOpcount_3/Read/ReadVariableOp!Hidden/kernel/Read/ReadVariableOpHidden/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_4/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_5/Read/ReadVariableOp(Adam/Hidden/kernel/m/Read/ReadVariableOp&Adam/Hidden/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp(Adam/Hidden/kernel/v/Read/ReadVariableOp&Adam/Hidden/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst_16*2
Tin+
)2'							*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_1648941
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameMutableHashTableMutableHashTable_1meanvariancecountmean_1
variance_1count_1mean_2
variance_2count_2mean_3
variance_3count_3Hidden/kernelHidden/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_4total_1count_5Adam/Hidden/kernel/mAdam/Hidden/bias/mAdam/Output/kernel/mAdam/Output/bias/mAdam/Hidden/kernel/vAdam/Hidden/bias/vAdam/Output/kernel/vAdam/Output/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_1649056??
?
o
6__inference_category_encoding_21_layer_call_fn_1648541

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
B__inference_Input_layer_call_and_return_conditional_losses_1648596
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
?
?
*__inference_model_10_layer_call_fn_1647995
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1647712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_model_10_layer_call_fn_1647789
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1647712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:ZV
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?B
?
E__inference_model_10_layer_call_and_return_conditional_losses_1647527

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x 
hidden_1647504:
hidden_1647506: 
output_1647521:
output_1647523:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_20/StatefulPartitionedCall?,category_encoding_21/StatefulPartitionedCall?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????m
normalization_20/subSubinputsnormalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_21/subSubinputs_1normalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_22/subSubinputs_2normalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_23/subSubinputs_3normalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:??????????
,category_encoding_20/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_20/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441?
,category_encoding_21/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_21/Identity:output:0-^category_encoding_20/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477?
Input/PartitionedCallPartitionedCallnormalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:05category_encoding_20/StatefulPartitionedCall:output:05category_encoding_21/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Input_layer_call_and_return_conditional_losses_1647490?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1647504hidden_1647506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1647521output_1647523*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1647520v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_20/StatefulPartitionedCall-^category_encoding_21/StatefulPartitionedCall/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_20/StatefulPartitionedCall,category_encoding_20/StatefulPartitionedCall2\
,category_encoding_21/StatefulPartitionedCall,category_encoding_21/StatefulPartitionedCall2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?	
?
B__inference_Input_layer_call_and_return_conditional_losses_1647490

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__destroyer_1648687
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441

inputs	
identity??Assert/AssertV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       C
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       E
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: K
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: J
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: W
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: O

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: ?
Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13h
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 T
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:h
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: h
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: d
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : q
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: [
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       W
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: `
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rf
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: Y
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: d
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rk
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: d
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Ro
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: c
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(n
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????V
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
0
 __inference__initializer_1648664
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_adapt_step_1648295
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
__inference_save_fn_1648748
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1648536

inputs	
identity??Assert/AssertV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       C
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       E
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: K
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: J
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: W
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: O

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: ?
Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13h
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 T
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:h
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: h
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: d
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : q
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: [
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       W
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: `
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rf
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: Y
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: d
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rk
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: d
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Ro
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: c
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(n
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????V
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
 __inference__initializer_1648682:
6key_value_init1622445_lookuptableimportv2_table_handle2
.key_value_init1622445_lookuptableimportv2_keys4
0key_value_init1622445_lookuptableimportv2_values	
identity??)key_value_init1622445/LookupTableImportV2?
)key_value_init1622445/LookupTableImportV2LookupTableImportV26key_value_init1622445_lookuptableimportv2_table_handle.key_value_init1622445_lookuptableimportv2_keys0key_value_init1622445_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init1622445/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1622445/LookupTableImportV2)key_value_init1622445/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
.
__inference__destroyer_1648669
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
<
__inference__creator_1648674
identity??
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1622446*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
??
?
#__inference__traced_restore_1649056
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: Q
Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1: 
assignvariableop_mean: %
assignvariableop_1_variance: "
assignvariableop_2_count:	 #
assignvariableop_3_mean_1: '
assignvariableop_4_variance_1: $
assignvariableop_5_count_1:	 #
assignvariableop_6_mean_2: '
assignvariableop_7_variance_2: $
assignvariableop_8_count_2:	 #
assignvariableop_9_mean_3: (
assignvariableop_10_variance_3: %
assignvariableop_11_count_3:	 3
!assignvariableop_12_hidden_kernel:-
assignvariableop_13_hidden_bias:3
!assignvariableop_14_output_kernel:-
assignvariableop_15_output_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: #
assignvariableop_21_total: %
assignvariableop_22_count_4: %
assignvariableop_23_total_1: %
assignvariableop_24_count_5: :
(assignvariableop_25_adam_hidden_kernel_m:4
&assignvariableop_26_adam_hidden_bias_m::
(assignvariableop_27_adam_output_kernel_m:4
&assignvariableop_28_adam_output_bias_m::
(assignvariableop_29_adam_hidden_kernel_v:4
&assignvariableop_30_adam_hidden_bias_v::
(assignvariableop_31_adam_output_kernel_v:4
&assignvariableop_32_adam_output_bias_v:
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?4MutableHashTable_table_restore_1/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&							?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtableRestoreV2:tensors:0RestoreV2:tensors:1*	
Tin0*

Tout0	*#
_class
loc:@MutableHashTable*
_output_shapes
 ?
4MutableHashTable_table_restore_1/LookupTableImportV2LookupTableImportV2Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1RestoreV2:tensors:2RestoreV2:tensors:3*	
Tin0*

Tout0	*%
_class
loc:@MutableHashTable_1*
_output_shapes
 [
IdentityIdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_mean_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_variance_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	^

Identity_6IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_mean_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_7IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_variance_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_8IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	^

Identity_9IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_mean_3Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_variance_3Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_hidden_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOpassignvariableop_13_hidden_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_output_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpassignvariableop_15_output_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_4Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_5Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp(assignvariableop_25_adam_hidden_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp&assignvariableop_26_adam_hidden_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_output_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_output_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_hidden_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_hidden_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_output_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_output_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_34Identity_34:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV22l
4MutableHashTable_table_restore_1/LookupTableImportV24MutableHashTable_table_restore_1/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_class
loc:@MutableHashTable:+'
%
_class
loc:@MutableHashTable_1
?
?
__inference_restore_fn_1648756
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
__inference_adapt_step_1648309
iterator

iterator_19
5none_lookup_table_find_lookuptablefindv2_table_handle:
6none_lookup_table_find_lookuptablefindv2_default_value	??IteratorGetNext?(None_lookup_table_find/LookupTableFindV2?,None_lookup_table_insert/LookupTableInsertV2?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2`
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????v
ReshapeReshapeIteratorGetNext:components:0Reshape/shape:output:0*
T0*#
_output_shapes
:??????????
UniqueWithCountsUniqueWithCountsReshape:output:0*
T0*A
_output_shapes/
-:?????????:?????????:?????????*
out_idx0	?
(None_lookup_table_find/LookupTableFindV2LookupTableFindV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:06none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
:|
addAddV2UniqueWithCounts:count:01None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:?
,None_lookup_table_insert/LookupTableInsertV2LookupTableInsertV25none_lookup_table_find_lookuptablefindv2_table_handleUniqueWithCounts:y:0add:z:0)^None_lookup_table_find/LookupTableFindV2",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 *(
_construction_contextkEagerRuntime*
_input_shapes

: : : : 2"
IteratorGetNextIteratorGetNext2T
(None_lookup_table_find/LookupTableFindV2(None_lookup_table_find/LookupTableFindV22\
,None_lookup_table_insert/LookupTableInsertV2,None_lookup_table_insert/LookupTableInsertV2:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator:

_output_shapes
: 
?
?
(__inference_Output_layer_call_fn_1648625

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1647520o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?B
?
E__inference_model_10_layer_call_and_return_conditional_losses_1647712

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x 
hidden_1647701:
hidden_1647703: 
output_1647706:
output_1647708:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_20/StatefulPartitionedCall?,category_encoding_21/StatefulPartitionedCall?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????m
normalization_20/subSubinputsnormalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_21/subSubinputs_1normalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_22/subSubinputs_2normalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_23/subSubinputs_3normalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:??????????
,category_encoding_20/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_20/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441?
,category_encoding_21/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_21/Identity:output:0-^category_encoding_20/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477?
Input/PartitionedCallPartitionedCallnormalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:05category_encoding_20/StatefulPartitionedCall:output:05category_encoding_21/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Input_layer_call_and_return_conditional_losses_1647490?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1647701hidden_1647703*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1647706output_1647708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1647520v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_20/StatefulPartitionedCall-^category_encoding_21/StatefulPartitionedCall/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_20/StatefulPartitionedCall,category_encoding_20/StatefulPartitionedCall2\
,category_encoding_21/StatefulPartitionedCall,category_encoding_21/StatefulPartitionedCall2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_model_10_layer_call_fn_1647562
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1647527o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:ZV
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
H
__inference__creator_1648659
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1622236*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?'
?
__inference_adapt_step_1648497
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(j
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	a
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
܊
?	
E__inference_model_10_layer_call_and_return_conditional_losses_1648116
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x7
%hidden_matmul_readvariableop_resource:4
&hidden_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Hidden/BiasAdd/ReadVariableOp?Hidden/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?"category_encoding_20/Assert/Assert?"category_encoding_21/Assert/Assert?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????o
normalization_20/subSubinputs_0normalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_21/subSubinputs_1normalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_22/subSubinputs_2normalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_23/subSubinputs_3normalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:?????????k
category_encoding_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_20/MaxMax"string_lookup_20/Identity:output:0#category_encoding_20/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_20/MinMin"string_lookup_20/Identity:output:0%category_encoding_20/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_20/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_20/CastCast$category_encoding_20/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_20/GreaterGreatercategory_encoding_20/Cast:y:0!category_encoding_20/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_20/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_20/Cast_1Cast&category_encoding_20/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_20/GreaterEqualGreaterEqual!category_encoding_20/Min:output:0category_encoding_20/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_20/LogicalAnd
LogicalAnd category_encoding_20/Greater:z:0%category_encoding_20/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_20/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
)category_encoding_20/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
"category_encoding_20/Assert/AssertAssert#category_encoding_20/LogicalAnd:z:02category_encoding_20/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
#category_encoding_20/bincount/ShapeShape"string_lookup_20/Identity:output:0#^category_encoding_20/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_20/bincount/ConstConst#^category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_20/bincount/ProdProd,category_encoding_20/bincount/Shape:output:0,category_encoding_20/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_20/bincount/Greater/yConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_20/bincount/GreaterGreater+category_encoding_20/bincount/Prod:output:00category_encoding_20/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_20/bincount/CastCast)category_encoding_20/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_20/bincount/Const_1Const#^category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_20/bincount/MaxMax"string_lookup_20/Identity:output:0.category_encoding_20/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_20/bincount/add/yConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_20/bincount/addAddV2*category_encoding_20/bincount/Max:output:0,category_encoding_20/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_20/bincount/mulMul&category_encoding_20/bincount/Cast:y:0%category_encoding_20/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_20/bincount/minlengthConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_20/bincount/MaximumMaximum0category_encoding_20/bincount/minlength:output:0%category_encoding_20/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_20/bincount/maxlengthConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_20/bincount/MinimumMinimum0category_encoding_20/bincount/maxlength:output:0)category_encoding_20/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_20/bincount/Const_2Const#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_20/bincount/DenseBincountDenseBincount"string_lookup_20/Identity:output:0)category_encoding_20/bincount/Minimum:z:0.category_encoding_20/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(k
category_encoding_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_21/MaxMax"string_lookup_21/Identity:output:0#category_encoding_21/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_21/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_21/MinMin"string_lookup_21/Identity:output:0%category_encoding_21/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_21/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_21/CastCast$category_encoding_21/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_21/GreaterGreatercategory_encoding_21/Cast:y:0!category_encoding_21/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_21/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_21/Cast_1Cast&category_encoding_21/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_21/GreaterEqualGreaterEqual!category_encoding_21/Min:output:0category_encoding_21/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_21/LogicalAnd
LogicalAnd category_encoding_21/Greater:z:0%category_encoding_21/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_21/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
)category_encoding_21/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
"category_encoding_21/Assert/AssertAssert#category_encoding_21/LogicalAnd:z:02category_encoding_21/Assert/Assert/data_0:output:0#^category_encoding_20/Assert/Assert*

T
2*
_output_shapes
 ?
#category_encoding_21/bincount/ShapeShape"string_lookup_21/Identity:output:0#^category_encoding_21/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_21/bincount/ConstConst#^category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_21/bincount/ProdProd,category_encoding_21/bincount/Shape:output:0,category_encoding_21/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_21/bincount/Greater/yConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_21/bincount/GreaterGreater+category_encoding_21/bincount/Prod:output:00category_encoding_21/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_21/bincount/CastCast)category_encoding_21/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_21/bincount/Const_1Const#^category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_21/bincount/MaxMax"string_lookup_21/Identity:output:0.category_encoding_21/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_21/bincount/add/yConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_21/bincount/addAddV2*category_encoding_21/bincount/Max:output:0,category_encoding_21/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_21/bincount/mulMul&category_encoding_21/bincount/Cast:y:0%category_encoding_21/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_21/bincount/minlengthConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_21/bincount/MaximumMaximum0category_encoding_21/bincount/minlength:output:0%category_encoding_21/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_21/bincount/maxlengthConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_21/bincount/MinimumMinimum0category_encoding_21/bincount/maxlength:output:0)category_encoding_21/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_21/bincount/Const_2Const#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_21/bincount/DenseBincountDenseBincount"string_lookup_21/Identity:output:0)category_encoding_21/bincount/Minimum:z:0.category_encoding_21/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(S
Input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
Input/concatConcatV2normalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:04category_encoding_20/bincount/DenseBincount:output:04category_encoding_21/bincount/DenseBincount:output:0Input/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
Hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Hidden/MatMulMatMulInput/concat:output:0$Hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Hidden/BiasAddBiasAddHidden/MatMul:product:0%Hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Hidden/ReluReluHidden/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output/MatMulMatMulHidden/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/BiasAdd/ReadVariableOp^Hidden/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp#^category_encoding_20/Assert/Assert#^category_encoding_21/Assert/Assert/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2>
Hidden/BiasAdd/ReadVariableOpHidden/BiasAdd/ReadVariableOp2<
Hidden/MatMul/ReadVariableOpHidden/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2H
"category_encoding_20/Assert/Assert"category_encoding_20/Assert/Assert2H
"category_encoding_21/Assert/Assert"category_encoding_21/Assert/Assert2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
*__inference_model_10_layer_call_fn_1647953
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_10_layer_call_and_return_conditional_losses_1647527o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
?
 __inference__initializer_1648649:
6key_value_init1622316_lookuptableimportv2_table_handle2
.key_value_init1622316_lookuptableimportv2_keys4
0key_value_init1622316_lookuptableimportv2_values	
identity??)key_value_init1622316/LookupTableImportV2?
)key_value_init1622316/LookupTableImportV2LookupTableImportV26key_value_init1622316_lookuptableimportv2_table_handle.key_value_init1622316_lookuptableimportv2_keys0key_value_init1622316_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init1622316/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1622316/LookupTableImportV2)key_value_init1622316/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
o
6__inference_category_encoding_20_layer_call_fn_1648502

inputs	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__destroyer_1648654
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
(__inference_Hidden_layer_call_fn_1648605

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_Input_layer_call_fn_1648585
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Input_layer_call_and_return_conditional_losses_1647490`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
?
?
__inference_<lambda>_1648764:
6key_value_init1622316_lookuptableimportv2_table_handle2
.key_value_init1622316_lookuptableimportv2_keys4
0key_value_init1622316_lookuptableimportv2_values	
identity??)key_value_init1622316/LookupTableImportV2?
)key_value_init1622316/LookupTableImportV2LookupTableImportV26key_value_init1622316_lookuptableimportv2_table_handle.key_value_init1622316_lookuptableimportv2_keys0key_value_init1622316_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init1622316/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1622316/LookupTableImportV2)key_value_init1622316/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
מ
?

"__inference__wrapped_model_1647354
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclassH
Dmodel_10_string_lookup_21_none_lookup_lookuptablefindv2_table_handleI
Emodel_10_string_lookup_21_none_lookup_lookuptablefindv2_default_value	H
Dmodel_10_string_lookup_20_none_lookup_lookuptablefindv2_table_handleI
Emodel_10_string_lookup_20_none_lookup_lookuptablefindv2_default_value	#
model_10_normalization_20_sub_y$
 model_10_normalization_20_sqrt_x#
model_10_normalization_21_sub_y$
 model_10_normalization_21_sqrt_x#
model_10_normalization_22_sub_y$
 model_10_normalization_22_sqrt_x#
model_10_normalization_23_sub_y$
 model_10_normalization_23_sqrt_x@
.model_10_hidden_matmul_readvariableop_resource:=
/model_10_hidden_biasadd_readvariableop_resource:@
.model_10_output_matmul_readvariableop_resource:=
/model_10_output_biasadd_readvariableop_resource:
identity??&model_10/Hidden/BiasAdd/ReadVariableOp?%model_10/Hidden/MatMul/ReadVariableOp?&model_10/Output/BiasAdd/ReadVariableOp?%model_10/Output/MatMul/ReadVariableOp?+model_10/category_encoding_20/Assert/Assert?+model_10/category_encoding_21/Assert/Assert?7model_10/string_lookup_20/None_Lookup/LookupTableFindV2?7model_10/string_lookup_21/None_Lookup/LookupTableFindV2?
7model_10/string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_10_string_lookup_21_none_lookup_lookuptablefindv2_table_handlespectralclassEmodel_10_string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
"model_10/string_lookup_21/IdentityIdentity@model_10/string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
7model_10/string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_10_string_lookup_20_none_lookup_lookuptablefindv2_table_handle	starcolorEmodel_10_string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
"model_10/string_lookup_20/IdentityIdentity@model_10/string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
model_10/normalization_20/subSubtemperaturemodel_10_normalization_20_sub_y*
T0*'
_output_shapes
:?????????q
model_10/normalization_20/SqrtSqrt model_10_normalization_20_sqrt_x*
T0*
_output_shapes

:h
#model_10/normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
!model_10/normalization_20/MaximumMaximum"model_10/normalization_20/Sqrt:y:0,model_10/normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
!model_10/normalization_20/truedivRealDiv!model_10/normalization_20/sub:z:0%model_10/normalization_20/Maximum:z:0*
T0*'
_output_shapes
:??????????
model_10/normalization_21/subSub
luminositymodel_10_normalization_21_sub_y*
T0*'
_output_shapes
:?????????q
model_10/normalization_21/SqrtSqrt model_10_normalization_21_sqrt_x*
T0*
_output_shapes

:h
#model_10/normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
!model_10/normalization_21/MaximumMaximum"model_10/normalization_21/Sqrt:y:0,model_10/normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
!model_10/normalization_21/truedivRealDiv!model_10/normalization_21/sub:z:0%model_10/normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????
model_10/normalization_22/subSubradiusmodel_10_normalization_22_sub_y*
T0*'
_output_shapes
:?????????q
model_10/normalization_22/SqrtSqrt model_10_normalization_22_sqrt_x*
T0*
_output_shapes

:h
#model_10/normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
!model_10/normalization_22/MaximumMaximum"model_10/normalization_22/Sqrt:y:0,model_10/normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
!model_10/normalization_22/truedivRealDiv!model_10/normalization_22/sub:z:0%model_10/normalization_22/Maximum:z:0*
T0*'
_output_shapes
:??????????
model_10/normalization_23/subSubabsolutemagnitudemodel_10_normalization_23_sub_y*
T0*'
_output_shapes
:?????????q
model_10/normalization_23/SqrtSqrt model_10_normalization_23_sqrt_x*
T0*
_output_shapes

:h
#model_10/normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
!model_10/normalization_23/MaximumMaximum"model_10/normalization_23/Sqrt:y:0,model_10/normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
!model_10/normalization_23/truedivRealDiv!model_10/normalization_23/sub:z:0%model_10/normalization_23/Maximum:z:0*
T0*'
_output_shapes
:?????????t
#model_10/category_encoding_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!model_10/category_encoding_20/MaxMax+model_10/string_lookup_20/Identity:output:0,model_10/category_encoding_20/Const:output:0*
T0	*
_output_shapes
: v
%model_10/category_encoding_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
!model_10/category_encoding_20/MinMin+model_10/string_lookup_20/Identity:output:0.model_10/category_encoding_20/Const_1:output:0*
T0	*
_output_shapes
: f
$model_10/category_encoding_20/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
"model_10/category_encoding_20/CastCast-model_10/category_encoding_20/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
%model_10/category_encoding_20/GreaterGreater&model_10/category_encoding_20/Cast:y:0*model_10/category_encoding_20/Max:output:0*
T0	*
_output_shapes
: h
&model_10/category_encoding_20/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_10/category_encoding_20/Cast_1Cast/model_10/category_encoding_20/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*model_10/category_encoding_20/GreaterEqualGreaterEqual*model_10/category_encoding_20/Min:output:0(model_10/category_encoding_20/Cast_1:y:0*
T0	*
_output_shapes
: ?
(model_10/category_encoding_20/LogicalAnd
LogicalAnd)model_10/category_encoding_20/Greater:z:0.model_10/category_encoding_20/GreaterEqual:z:0*
_output_shapes
: ?
*model_10/category_encoding_20/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
2model_10/category_encoding_20/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
+model_10/category_encoding_20/Assert/AssertAssert,model_10/category_encoding_20/LogicalAnd:z:0;model_10/category_encoding_20/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
,model_10/category_encoding_20/bincount/ShapeShape+model_10/string_lookup_20/Identity:output:0,^model_10/category_encoding_20/Assert/Assert*
T0	*
_output_shapes
:?
,model_10/category_encoding_20/bincount/ConstConst,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
+model_10/category_encoding_20/bincount/ProdProd5model_10/category_encoding_20/bincount/Shape:output:05model_10/category_encoding_20/bincount/Const:output:0*
T0*
_output_shapes
: ?
0model_10/category_encoding_20/bincount/Greater/yConst,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
.model_10/category_encoding_20/bincount/GreaterGreater4model_10/category_encoding_20/bincount/Prod:output:09model_10/category_encoding_20/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
+model_10/category_encoding_20/bincount/CastCast2model_10/category_encoding_20/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
.model_10/category_encoding_20/bincount/Const_1Const,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
*model_10/category_encoding_20/bincount/MaxMax+model_10/string_lookup_20/Identity:output:07model_10/category_encoding_20/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
,model_10/category_encoding_20/bincount/add/yConst,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
*model_10/category_encoding_20/bincount/addAddV23model_10/category_encoding_20/bincount/Max:output:05model_10/category_encoding_20/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
*model_10/category_encoding_20/bincount/mulMul/model_10/category_encoding_20/bincount/Cast:y:0.model_10/category_encoding_20/bincount/add:z:0*
T0	*
_output_shapes
: ?
0model_10/category_encoding_20/bincount/minlengthConst,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
.model_10/category_encoding_20/bincount/MaximumMaximum9model_10/category_encoding_20/bincount/minlength:output:0.model_10/category_encoding_20/bincount/mul:z:0*
T0	*
_output_shapes
: ?
0model_10/category_encoding_20/bincount/maxlengthConst,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
.model_10/category_encoding_20/bincount/MinimumMinimum9model_10/category_encoding_20/bincount/maxlength:output:02model_10/category_encoding_20/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
.model_10/category_encoding_20/bincount/Const_2Const,^model_10/category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
4model_10/category_encoding_20/bincount/DenseBincountDenseBincount+model_10/string_lookup_20/Identity:output:02model_10/category_encoding_20/bincount/Minimum:z:07model_10/category_encoding_20/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(t
#model_10/category_encoding_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
!model_10/category_encoding_21/MaxMax+model_10/string_lookup_21/Identity:output:0,model_10/category_encoding_21/Const:output:0*
T0	*
_output_shapes
: v
%model_10/category_encoding_21/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
!model_10/category_encoding_21/MinMin+model_10/string_lookup_21/Identity:output:0.model_10/category_encoding_21/Const_1:output:0*
T0	*
_output_shapes
: f
$model_10/category_encoding_21/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
"model_10/category_encoding_21/CastCast-model_10/category_encoding_21/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
%model_10/category_encoding_21/GreaterGreater&model_10/category_encoding_21/Cast:y:0*model_10/category_encoding_21/Max:output:0*
T0	*
_output_shapes
: h
&model_10/category_encoding_21/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
$model_10/category_encoding_21/Cast_1Cast/model_10/category_encoding_21/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
*model_10/category_encoding_21/GreaterEqualGreaterEqual*model_10/category_encoding_21/Min:output:0(model_10/category_encoding_21/Cast_1:y:0*
T0	*
_output_shapes
: ?
(model_10/category_encoding_21/LogicalAnd
LogicalAnd)model_10/category_encoding_21/Greater:z:0.model_10/category_encoding_21/GreaterEqual:z:0*
_output_shapes
: ?
*model_10/category_encoding_21/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
2model_10/category_encoding_21/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
+model_10/category_encoding_21/Assert/AssertAssert,model_10/category_encoding_21/LogicalAnd:z:0;model_10/category_encoding_21/Assert/Assert/data_0:output:0,^model_10/category_encoding_20/Assert/Assert*

T
2*
_output_shapes
 ?
,model_10/category_encoding_21/bincount/ShapeShape+model_10/string_lookup_21/Identity:output:0,^model_10/category_encoding_21/Assert/Assert*
T0	*
_output_shapes
:?
,model_10/category_encoding_21/bincount/ConstConst,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
+model_10/category_encoding_21/bincount/ProdProd5model_10/category_encoding_21/bincount/Shape:output:05model_10/category_encoding_21/bincount/Const:output:0*
T0*
_output_shapes
: ?
0model_10/category_encoding_21/bincount/Greater/yConst,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
.model_10/category_encoding_21/bincount/GreaterGreater4model_10/category_encoding_21/bincount/Prod:output:09model_10/category_encoding_21/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
+model_10/category_encoding_21/bincount/CastCast2model_10/category_encoding_21/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
.model_10/category_encoding_21/bincount/Const_1Const,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
*model_10/category_encoding_21/bincount/MaxMax+model_10/string_lookup_21/Identity:output:07model_10/category_encoding_21/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
,model_10/category_encoding_21/bincount/add/yConst,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
*model_10/category_encoding_21/bincount/addAddV23model_10/category_encoding_21/bincount/Max:output:05model_10/category_encoding_21/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
*model_10/category_encoding_21/bincount/mulMul/model_10/category_encoding_21/bincount/Cast:y:0.model_10/category_encoding_21/bincount/add:z:0*
T0	*
_output_shapes
: ?
0model_10/category_encoding_21/bincount/minlengthConst,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
.model_10/category_encoding_21/bincount/MaximumMaximum9model_10/category_encoding_21/bincount/minlength:output:0.model_10/category_encoding_21/bincount/mul:z:0*
T0	*
_output_shapes
: ?
0model_10/category_encoding_21/bincount/maxlengthConst,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
.model_10/category_encoding_21/bincount/MinimumMinimum9model_10/category_encoding_21/bincount/maxlength:output:02model_10/category_encoding_21/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
.model_10/category_encoding_21/bincount/Const_2Const,^model_10/category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
4model_10/category_encoding_21/bincount/DenseBincountDenseBincount+model_10/string_lookup_21/Identity:output:02model_10/category_encoding_21/bincount/Minimum:z:07model_10/category_encoding_21/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(\
model_10/Input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_10/Input/concatConcatV2%model_10/normalization_20/truediv:z:0%model_10/normalization_21/truediv:z:0%model_10/normalization_22/truediv:z:0%model_10/normalization_23/truediv:z:0=model_10/category_encoding_20/bincount/DenseBincount:output:0=model_10/category_encoding_21/bincount/DenseBincount:output:0#model_10/Input/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
%model_10/Hidden/MatMul/ReadVariableOpReadVariableOp.model_10_hidden_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_10/Hidden/MatMulMatMulmodel_10/Input/concat:output:0-model_10/Hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_10/Hidden/BiasAdd/ReadVariableOpReadVariableOp/model_10_hidden_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_10/Hidden/BiasAddBiasAdd model_10/Hidden/MatMul:product:0.model_10/Hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
model_10/Hidden/ReluRelu model_10/Hidden/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
%model_10/Output/MatMul/ReadVariableOpReadVariableOp.model_10_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_10/Output/MatMulMatMul"model_10/Hidden/Relu:activations:0-model_10/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
&model_10/Output/BiasAdd/ReadVariableOpReadVariableOp/model_10_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_10/Output/BiasAddBiasAdd model_10/Output/MatMul:product:0.model_10/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
model_10/Output/SoftmaxSoftmax model_10/Output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????p
IdentityIdentity!model_10/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^model_10/Hidden/BiasAdd/ReadVariableOp&^model_10/Hidden/MatMul/ReadVariableOp'^model_10/Output/BiasAdd/ReadVariableOp&^model_10/Output/MatMul/ReadVariableOp,^model_10/category_encoding_20/Assert/Assert,^model_10/category_encoding_21/Assert/Assert8^model_10/string_lookup_20/None_Lookup/LookupTableFindV28^model_10/string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2P
&model_10/Hidden/BiasAdd/ReadVariableOp&model_10/Hidden/BiasAdd/ReadVariableOp2N
%model_10/Hidden/MatMul/ReadVariableOp%model_10/Hidden/MatMul/ReadVariableOp2P
&model_10/Output/BiasAdd/ReadVariableOp&model_10/Output/BiasAdd/ReadVariableOp2N
%model_10/Output/MatMul/ReadVariableOp%model_10/Output/MatMul/ReadVariableOp2Z
+model_10/category_encoding_20/Assert/Assert+model_10/category_encoding_20/Assert/Assert2Z
+model_10/category_encoding_21/Assert/Assert+model_10/category_encoding_21/Assert/Assert2r
7model_10/string_lookup_20/None_Lookup/LookupTableFindV27model_10/string_lookup_20/None_Lookup/LookupTableFindV22r
7model_10/string_lookup_21/None_Lookup/LookupTableFindV27model_10/string_lookup_21/None_Lookup/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:ZV
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?C
?
E__inference_model_10_layer_call_and_return_conditional_losses_1647905
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x 
hidden_1647894:
hidden_1647896: 
output_1647899:
output_1647901:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_20/StatefulPartitionedCall?,category_encoding_21/StatefulPartitionedCall?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handlespectralclass<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handle	starcolor<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????r
normalization_20/subSubtemperaturenormalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????q
normalization_21/subSub
luminositynormalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_22/subSubradiusnormalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????x
normalization_23/subSubabsolutemagnitudenormalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:??????????
,category_encoding_20/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_20/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441?
,category_encoding_21/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_21/Identity:output:0-^category_encoding_20/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477?
Input/PartitionedCallPartitionedCallnormalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:05category_encoding_20/StatefulPartitionedCall:output:05category_encoding_21/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Input_layer_call_and_return_conditional_losses_1647490?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1647894hidden_1647896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1647899output_1647901*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1647520v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_20/StatefulPartitionedCall-^category_encoding_21/StatefulPartitionedCall/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_20/StatefulPartitionedCall,category_encoding_20/StatefulPartitionedCall2\
,category_encoding_21/StatefulPartitionedCall,category_encoding_21/StatefulPartitionedCall2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:ZV
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?

?
C__inference_Output_layer_call_and_return_conditional_losses_1648636

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
?
__inference_adapt_step_1648403
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(j
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	a
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
__inference_save_fn_1648721
checkpoint_keyP
Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	???MutableHashTable_lookup_table_export_values/LookupTableExportV2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Lmutablehashtable_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*
_output_shapes

::K
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keysK
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: O
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-valuesO
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: E
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
: F
ConstConst*
_output_shapes
: *
dtype0*
valueB B N

Identity_1IdentityConst:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_2IdentityFMutableHashTable_lookup_table_export_values/LookupTableExportV2:keys:0^NoOp*
T0*
_output_shapes
:I

Identity_3Identity	add_1:z:0^NoOp*
T0*
_output_shapes
: H
Const_1Const*
_output_shapes
: *
dtype0*
valueB B P

Identity_4IdentityConst_1:output:0^NoOp*
T0*
_output_shapes
: ?

Identity_5IdentityHMutableHashTable_lookup_table_export_values/LookupTableExportV2:values:0^NoOp*
T0	*
_output_shapes
:?
NoOpNoOp@^MutableHashTable_lookup_table_export_values/LookupTableExportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2?
?MutableHashTable_lookup_table_export_values/LookupTableExportV2?MutableHashTable_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
0
 __inference__initializer_1648697
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?

?
C__inference_Hidden_layer_call_and_return_conditional_losses_1648616

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
.
__inference__destroyer_1648702
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?C
?
E__inference_model_10_layer_call_and_return_conditional_losses_1647847
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x 
hidden_1647836:
hidden_1647838: 
output_1647841:
output_1647843:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_20/StatefulPartitionedCall?,category_encoding_21/StatefulPartitionedCall?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handlespectralclass<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handle	starcolor<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????r
normalization_20/subSubtemperaturenormalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????q
normalization_21/subSub
luminositynormalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????m
normalization_22/subSubradiusnormalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????x
normalization_23/subSubabsolutemagnitudenormalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:??????????
,category_encoding_20/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_20/Identity:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1647441?
,category_encoding_21/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_21/Identity:output:0-^category_encoding_20/StatefulPartitionedCall*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477?
Input/PartitionedCallPartitionedCallnormalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:05category_encoding_20/StatefulPartitionedCall:output:05category_encoding_21/StatefulPartitionedCall:output:0*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Input_layer_call_and_return_conditional_losses_1647490?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1647836hidden_1647838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Hidden_layer_call_and_return_conditional_losses_1647503?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1647841output_1647843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_Output_layer_call_and_return_conditional_losses_1647520v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_20/StatefulPartitionedCall-^category_encoding_21/StatefulPartitionedCall/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_20/StatefulPartitionedCall,category_encoding_20/StatefulPartitionedCall2\
,category_encoding_21/StatefulPartitionedCall,category_encoding_21/StatefulPartitionedCall2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:T P
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:ZV
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?
<
__inference__creator_1648641
identity??
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1622317*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
?
H
__inference__creator_1648692
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1622365*
value_dtype0	]
IdentityIdentityMutableHashTable:table_handle:0^NoOp*
T0*
_output_shapes
: Y
NoOpNoOp^MutableHashTable*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2$
MutableHashTableMutableHashTable
?'
?
__inference_adapt_step_1648356
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2	k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(j
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	a
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
,
__inference_<lambda>_1648769
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1648575

inputs	
identity??Assert/AssertV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       C
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       E
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: K
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: J
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: W
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: O

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: ?
Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8h
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 T
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:h
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: h
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: d
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : q
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: [
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       W
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: `
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rf
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: Y
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: d
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rk
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: d
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Ro
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: c
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(n
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????V
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
܊
?	
E__inference_model_10_layer_call_and_return_conditional_losses_1648237
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_21_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_21_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_20_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_20_none_lookup_lookuptablefindv2_default_value	
normalization_20_sub_y
normalization_20_sqrt_x
normalization_21_sub_y
normalization_21_sqrt_x
normalization_22_sub_y
normalization_22_sqrt_x
normalization_23_sub_y
normalization_23_sqrt_x7
%hidden_matmul_readvariableop_resource:4
&hidden_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Hidden/BiasAdd/ReadVariableOp?Hidden/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?"category_encoding_20/Assert/Assert?"category_encoding_21/Assert/Assert?.string_lookup_20/None_Lookup/LookupTableFindV2?.string_lookup_21/None_Lookup/LookupTableFindV2?
.string_lookup_21/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_21_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_21_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_21/IdentityIdentity7string_lookup_21/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_20/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_20_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_20_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_20/IdentityIdentity7string_lookup_20/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????o
normalization_20/subSubinputs_0normalization_20_sub_y*
T0*'
_output_shapes
:?????????_
normalization_20/SqrtSqrtnormalization_20_sqrt_x*
T0*
_output_shapes

:_
normalization_20/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_20/MaximumMaximumnormalization_20/Sqrt:y:0#normalization_20/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_20/truedivRealDivnormalization_20/sub:z:0normalization_20/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_21/subSubinputs_1normalization_21_sub_y*
T0*'
_output_shapes
:?????????_
normalization_21/SqrtSqrtnormalization_21_sqrt_x*
T0*
_output_shapes

:_
normalization_21/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_21/MaximumMaximumnormalization_21/Sqrt:y:0#normalization_21/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_21/truedivRealDivnormalization_21/sub:z:0normalization_21/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_22/subSubinputs_2normalization_22_sub_y*
T0*'
_output_shapes
:?????????_
normalization_22/SqrtSqrtnormalization_22_sqrt_x*
T0*
_output_shapes

:_
normalization_22/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_22/MaximumMaximumnormalization_22/Sqrt:y:0#normalization_22/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_22/truedivRealDivnormalization_22/sub:z:0normalization_22/Maximum:z:0*
T0*'
_output_shapes
:?????????o
normalization_23/subSubinputs_3normalization_23_sub_y*
T0*'
_output_shapes
:?????????_
normalization_23/SqrtSqrtnormalization_23_sqrt_x*
T0*
_output_shapes

:_
normalization_23/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3?
normalization_23/MaximumMaximumnormalization_23/Sqrt:y:0#normalization_23/Maximum/y:output:0*
T0*
_output_shapes

:?
normalization_23/truedivRealDivnormalization_23/sub:z:0normalization_23/Maximum:z:0*
T0*'
_output_shapes
:?????????k
category_encoding_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_20/MaxMax"string_lookup_20/Identity:output:0#category_encoding_20/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_20/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_20/MinMin"string_lookup_20/Identity:output:0%category_encoding_20/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_20/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_20/CastCast$category_encoding_20/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_20/GreaterGreatercategory_encoding_20/Cast:y:0!category_encoding_20/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_20/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_20/Cast_1Cast&category_encoding_20/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_20/GreaterEqualGreaterEqual!category_encoding_20/Min:output:0category_encoding_20/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_20/LogicalAnd
LogicalAnd category_encoding_20/Greater:z:0%category_encoding_20/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_20/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
)category_encoding_20/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
"category_encoding_20/Assert/AssertAssert#category_encoding_20/LogicalAnd:z:02category_encoding_20/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
#category_encoding_20/bincount/ShapeShape"string_lookup_20/Identity:output:0#^category_encoding_20/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_20/bincount/ConstConst#^category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_20/bincount/ProdProd,category_encoding_20/bincount/Shape:output:0,category_encoding_20/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_20/bincount/Greater/yConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_20/bincount/GreaterGreater+category_encoding_20/bincount/Prod:output:00category_encoding_20/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_20/bincount/CastCast)category_encoding_20/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_20/bincount/Const_1Const#^category_encoding_20/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_20/bincount/MaxMax"string_lookup_20/Identity:output:0.category_encoding_20/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_20/bincount/add/yConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_20/bincount/addAddV2*category_encoding_20/bincount/Max:output:0,category_encoding_20/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_20/bincount/mulMul&category_encoding_20/bincount/Cast:y:0%category_encoding_20/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_20/bincount/minlengthConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_20/bincount/MaximumMaximum0category_encoding_20/bincount/minlength:output:0%category_encoding_20/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_20/bincount/maxlengthConst#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_20/bincount/MinimumMinimum0category_encoding_20/bincount/maxlength:output:0)category_encoding_20/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_20/bincount/Const_2Const#^category_encoding_20/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_20/bincount/DenseBincountDenseBincount"string_lookup_20/Identity:output:0)category_encoding_20/bincount/Minimum:z:0.category_encoding_20/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(k
category_encoding_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_21/MaxMax"string_lookup_21/Identity:output:0#category_encoding_21/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_21/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_21/MinMin"string_lookup_21/Identity:output:0%category_encoding_21/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_21/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_21/CastCast$category_encoding_21/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_21/GreaterGreatercategory_encoding_21/Cast:y:0!category_encoding_21/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_21/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_21/Cast_1Cast&category_encoding_21/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_21/GreaterEqualGreaterEqual!category_encoding_21/Min:output:0category_encoding_21/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_21/LogicalAnd
LogicalAnd category_encoding_21/Greater:z:0%category_encoding_21/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_21/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
)category_encoding_21/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
"category_encoding_21/Assert/AssertAssert#category_encoding_21/LogicalAnd:z:02category_encoding_21/Assert/Assert/data_0:output:0#^category_encoding_20/Assert/Assert*

T
2*
_output_shapes
 ?
#category_encoding_21/bincount/ShapeShape"string_lookup_21/Identity:output:0#^category_encoding_21/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_21/bincount/ConstConst#^category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_21/bincount/ProdProd,category_encoding_21/bincount/Shape:output:0,category_encoding_21/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_21/bincount/Greater/yConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_21/bincount/GreaterGreater+category_encoding_21/bincount/Prod:output:00category_encoding_21/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_21/bincount/CastCast)category_encoding_21/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_21/bincount/Const_1Const#^category_encoding_21/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_21/bincount/MaxMax"string_lookup_21/Identity:output:0.category_encoding_21/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_21/bincount/add/yConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_21/bincount/addAddV2*category_encoding_21/bincount/Max:output:0,category_encoding_21/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_21/bincount/mulMul&category_encoding_21/bincount/Cast:y:0%category_encoding_21/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_21/bincount/minlengthConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_21/bincount/MaximumMaximum0category_encoding_21/bincount/minlength:output:0%category_encoding_21/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_21/bincount/maxlengthConst#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_21/bincount/MinimumMinimum0category_encoding_21/bincount/maxlength:output:0)category_encoding_21/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_21/bincount/Const_2Const#^category_encoding_21/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_21/bincount/DenseBincountDenseBincount"string_lookup_21/Identity:output:0)category_encoding_21/bincount/Minimum:z:0.category_encoding_21/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(S
Input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
Input/concatConcatV2normalization_20/truediv:z:0normalization_21/truediv:z:0normalization_22/truediv:z:0normalization_23/truediv:z:04category_encoding_20/bincount/DenseBincount:output:04category_encoding_21/bincount/DenseBincount:output:0Input/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
Hidden/MatMul/ReadVariableOpReadVariableOp%hidden_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Hidden/MatMulMatMulInput/concat:output:0$Hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Hidden/BiasAdd/ReadVariableOpReadVariableOp&hidden_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Hidden/BiasAddBiasAddHidden/MatMul:product:0%Hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Hidden/ReluReluHidden/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output/MatMulMatMulHidden/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
Output/SoftmaxSoftmaxOutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????g
IdentityIdentityOutput/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/BiasAdd/ReadVariableOp^Hidden/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp#^category_encoding_20/Assert/Assert#^category_encoding_21/Assert/Assert/^string_lookup_20/None_Lookup/LookupTableFindV2/^string_lookup_21/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 2>
Hidden/BiasAdd/ReadVariableOpHidden/BiasAdd/ReadVariableOp2<
Hidden/MatMul/ReadVariableOpHidden/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2H
"category_encoding_20/Assert/Assert"category_encoding_20/Assert/Assert2H
"category_encoding_21/Assert/Assert"category_encoding_21/Assert/Assert2`
.string_lookup_20/None_Lookup/LookupTableFindV2.string_lookup_20/None_Lookup/LookupTableFindV22`
.string_lookup_21/None_Lookup/LookupTableFindV2.string_lookup_21/None_Lookup/LookupTableFindV2:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:
?'
?
__inference_adapt_step_1648450
iterator

iterator_1%
add_readvariableop_resource:	 !
readvariableop_resource: #
readvariableop_2_resource: ??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:?????????*&
output_shapes
:?????????*
output_types
2k
CastCastIteratorGetNext:components:0*

DstT0*

SrcT0*'
_output_shapes
:?????????o
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/meanMeanCast:y:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:?
moments/SquaredDifferenceSquaredDifferenceCast:y:0moments/StopGradient:output:0*
T0*'
_output_shapes
:?????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(j
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 M
ShapeShapeCast:y:0*
T0*
_output_shapes
:*
out_type0	a
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB"       O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: M
Cast_1CastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_2Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: K
truedivRealDiv
Cast_1:y:0
Cast_2:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: ^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0L
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
: T
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
: C
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
: `
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0R
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
: J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @F
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
: b
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
: *
dtype0R
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
: A
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
: R
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
: L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
: V
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
: E
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
: E
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
: ?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?H
?
 __inference__traced_save_1648941
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1	#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	%
!savev2_mean_1_read_readvariableop)
%savev2_variance_1_read_readvariableop&
"savev2_count_1_read_readvariableop	%
!savev2_mean_2_read_readvariableop)
%savev2_variance_2_read_readvariableop&
"savev2_count_2_read_readvariableop	%
!savev2_mean_3_read_readvariableop)
%savev2_variance_3_read_readvariableop&
"savev2_count_3_read_readvariableop	,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_4_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_5_read_readvariableop3
/savev2_adam_hidden_kernel_m_read_readvariableop1
-savev2_adam_hidden_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop3
/savev2_adam_hidden_kernel_v_read_readvariableop1
-savev2_adam_hidden_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const_16

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB4layer_with_weights-2/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-2/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-3/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-4/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-5/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop!savev2_mean_1_read_readvariableop%savev2_variance_1_read_readvariableop"savev2_count_1_read_readvariableop!savev2_mean_2_read_readvariableop%savev2_variance_2_read_readvariableop"savev2_count_2_read_readvariableop!savev2_mean_3_read_readvariableop%savev2_variance_3_read_readvariableop"savev2_count_3_read_readvariableop(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_4_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_5_read_readvariableop/savev2_adam_hidden_kernel_m_read_readvariableop-savev2_adam_hidden_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop/savev2_adam_hidden_kernel_v_read_readvariableop-savev2_adam_hidden_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const_16"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&							?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::: : : : : : : : : : : : ::::: : : : : : : : : ::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::&

_output_shapes
: 
?
?
__inference_restore_fn_1648729
restored_tensors_0
restored_tensors_1	C
?mutablehashtable_table_restore_lookuptableimportv2_table_handle
identity??2MutableHashTable_table_restore/LookupTableImportV2?
2MutableHashTable_table_restore/LookupTableImportV2LookupTableImportV2?mutablehashtable_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: {
NoOpNoOp3^MutableHashTable_table_restore/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

::: 2h
2MutableHashTable_table_restore/LookupTableImportV22MutableHashTable_table_restore/LookupTableImportV2:L H

_output_shapes
:
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
,
__inference_<lambda>_1648782
identityJ
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
?
?
__inference_<lambda>_1648777:
6key_value_init1622445_lookuptableimportv2_table_handle2
.key_value_init1622445_lookuptableimportv2_keys4
0key_value_init1622445_lookuptableimportv2_values	
identity??)key_value_init1622445/LookupTableImportV2?
)key_value_init1622445/LookupTableImportV2LookupTableImportV26key_value_init1622445_lookuptableimportv2_table_handle.key_value_init1622445_lookuptableimportv2_keys0key_value_init1622445_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: r
NoOpNoOp*^key_value_init1622445/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1622445/LookupTableImportV2)key_value_init1622445/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1647477

inputs	
identity??Assert/AssertV
ConstConst*
_output_shapes
:*
dtype0*
valueB"       C
MaxMaxinputsConst:output:0*
T0	*
_output_shapes
: X
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       E
MinMininputsConst_1:output:0*
T0	*
_output_shapes
: H
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: K
GreaterGreaterCast:y:0Max:output:0*
T0	*
_output_shapes
: J
Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : Q
Cast_1CastCast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: W
GreaterEqualGreaterEqualMin:output:0
Cast_1:y:0*
T0	*
_output_shapes
: O

LogicalAnd
LogicalAndGreater:z:0GreaterEqual:z:0*
_output_shapes
: ?
Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8h
Assert/AssertAssertLogicalAnd:z:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 T
bincount/ShapeShapeinputs^Assert/Assert*
T0	*
_output_shapes
:h
bincount/ConstConst^Assert/Assert*
_output_shapes
:*
dtype0*
valueB: h
bincount/ProdProdbincount/Shape:output:0bincount/Const:output:0*
T0*
_output_shapes
: d
bincount/Greater/yConst^Assert/Assert*
_output_shapes
: *
dtype0*
value	B : q
bincount/GreaterGreaterbincount/Prod:output:0bincount/Greater/y:output:0*
T0*
_output_shapes
: [
bincount/CastCastbincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
bincount/Const_1Const^Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       W
bincount/MaxMaxinputsbincount/Const_1:output:0*
T0	*
_output_shapes
: `
bincount/add/yConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rf
bincount/addAddV2bincount/Max:output:0bincount/add/y:output:0*
T0	*
_output_shapes
: Y
bincount/mulMulbincount/Cast:y:0bincount/add:z:0*
T0	*
_output_shapes
: d
bincount/minlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Rk
bincount/MaximumMaximumbincount/minlength:output:0bincount/mul:z:0*
T0	*
_output_shapes
: d
bincount/maxlengthConst^Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 Ro
bincount/MinimumMinimumbincount/maxlength:output:0bincount/Maximum:z:0*
T0	*
_output_shapes
: c
bincount/Const_2Const^Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
bincount/DenseBincountDenseBincountinputsbincount/Minimum:z:0bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(n
IdentityIdentitybincount/DenseBincount:output:0^NoOp*
T0*'
_output_shapes
:?????????V
NoOpNoOp^Assert/Assert*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????2
Assert/AssertAssert/Assert:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_Output_layer_call_and_return_conditional_losses_1647520

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_1648281
absolutemagnitude

luminosity

radius
spectralclass
	starcolor
temperature
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:

unknown_12:

unknown_13:

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*!
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_1647354o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : ::::::::: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
'
_output_shapes
:?????????
+
_user_specified_nameAbsoluteMagnitude:SO
'
_output_shapes
:?????????
$
_user_specified_name
Luminosity:OK
'
_output_shapes
:?????????
 
_user_specified_nameRadius:VR
'
_output_shapes
:?????????
'
_user_specified_nameSpectralClass:RN
'
_output_shapes
:?????????
#
_user_specified_name	StarColor:TP
'
_output_shapes
:?????????
%
_user_specified_nameTemperature:

_output_shapes
: :	

_output_shapes
: :$
 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

::$ 

_output_shapes

:"?L
saver_filename:0StatefulPartitionedCall_3:0StatefulPartitionedCall_48"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
AbsoluteMagnitude:
#serving_default_AbsoluteMagnitude:0?????????
A

Luminosity3
serving_default_Luminosity:0?????????
9
Radius/
serving_default_Radius:0?????????
G
SpectralClass6
serving_default_SpectralClass:0?????????
?
	StarColor2
serving_default_StarColor:0?????????
C
Temperature4
serving_default_Temperature:0?????????<
Output2
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-0
layer-6
layer_with_weights-1
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
a
lookup_table
token_counts
	keras_api
_adapt_function"
_tf_keras_layer
a
lookup_table
 token_counts
!	keras_api
"_adapt_function"
_tf_keras_layer
?
#
_keep_axis
$_reduce_axis
%_reduce_axis_mask
&_broadcast_shape
'mean
'
adapt_mean
(variance
(adapt_variance
	)count
*	keras_api
+_adapt_function"
_tf_keras_layer
?
,
_keep_axis
-_reduce_axis
._reduce_axis_mask
/_broadcast_shape
0mean
0
adapt_mean
1variance
1adapt_variance
	2count
3	keras_api
4_adapt_function"
_tf_keras_layer
?
5
_keep_axis
6_reduce_axis
7_reduce_axis_mask
8_broadcast_shape
9mean
9
adapt_mean
:variance
:adapt_variance
	;count
<	keras_api
=_adapt_function"
_tf_keras_layer
?
>
_keep_axis
?_reduce_axis
@_reduce_axis_mask
A_broadcast_shape
Bmean
B
adapt_mean
Cvariance
Cadapt_variance
	Dcount
E	keras_api
F_adapt_function"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Ykernel
Zbias
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?

akernel
bbias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses"
_tf_keras_layer
?
iiter

jbeta_1

kbeta_2
	ldecay
mlearning_rateYm?Zm?am?bm?Yv?Zv?av?bv?"
	optimizer
?
'2
(3
)4
05
16
27
98
:9
;10
B11
C12
D13
Y14
Z15
a16
b17"
trackable_list_wrapper
<
Y0
Z1
a2
b3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
*__inference_model_10_layer_call_fn_1647562
*__inference_model_10_layer_call_fn_1647953
*__inference_model_10_layer_call_fn_1647995
*__inference_model_10_layer_call_fn_1647789?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_10_layer_call_and_return_conditional_losses_1648116
E__inference_model_10_layer_call_and_return_conditional_losses_1648237
E__inference_model_10_layer_call_and_return_conditional_losses_1647847
E__inference_model_10_layer_call_and_return_conditional_losses_1647905?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1647354Temperature
LuminosityRadiusAbsoluteMagnitude	StarColorSpectralClass"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
sserving_default"
signature_map
j
t_initializer
u_create_resource
v_initialize
w_destroy_resourceR jCustom.StaticHashTable
Q
x_create_resource
y_initialize
z_destroy_resourceR Z
table??
"
_generic_user_object
?2?
__inference_adapt_step_1648295?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
j
{_initializer
|_create_resource
}_initialize
~_destroy_resourceR jCustom.StaticHashTable
S
_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
?2?
__inference_adapt_step_1648309?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_1648356?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_1648403?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_1648450?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
 2mean
: 2variance
:	 2count
"
_generic_user_object
?2?
__inference_adapt_step_1648497?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_category_encoding_20_layer_call_fn_1648502?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1648536?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_category_encoding_21_layer_call_fn_1648541?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1648575?
???
FullArgSpec.
args&?#
jself
jinputs
jcount_weights
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_Input_layer_call_fn_1648585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Input_layer_call_and_return_conditional_losses_1648596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:2Hidden/kernel
:2Hidden/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Hidden_layer_call_fn_1648605?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Hidden_layer_call_and_return_conditional_losses_1648616?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:2Output/kernel
:2Output/bias
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Output_layer_call_fn_1648625?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_Output_layer_call_and_return_conditional_losses_1648636?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
x
'2
(3
)4
05
16
27
98
:9
;10
B11
C12
D13"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_1648281AbsoluteMagnitude
LuminosityRadiusSpectralClass	StarColorTemperature"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
"
_generic_user_object
?2?
__inference__creator_1648641?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_1648649?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1648654?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_1648659?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_1648664?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1648669?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
"
_generic_user_object
?2?
__inference__creator_1648674?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_1648682?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1648687?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_1648692?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
 __inference__initializer_1648697?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_1648702?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
$:"2Adam/Hidden/kernel/m
:2Adam/Hidden/bias/m
$:"2Adam/Output/kernel/m
:2Adam/Output/bias/m
$:"2Adam/Hidden/kernel/v
:2Adam/Hidden/bias/v
$:"2Adam/Output/kernel/v
:2Adam/Output/bias/v
?B?
__inference_save_fn_1648721checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_1648729restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
?B?
__inference_save_fn_1648748checkpoint_key"?
???
FullArgSpec
args?
jcheckpoint_key
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?	
? 
?B?
__inference_restore_fn_1648756restored_tensors_0restored_tensors_1"?
???
FullArgSpec
args? 
varargsjrestored_tensors
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *?
	?
	?	
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15?
C__inference_Hidden_layer_call_and_return_conditional_losses_1648616\YZ/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_Hidden_layer_call_fn_1648605OYZ/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_Input_layer_call_and_return_conditional_losses_1648596????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "%?"
?
0?????????
? ?
'__inference_Input_layer_call_fn_1648585????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "???????????
C__inference_Output_layer_call_and_return_conditional_losses_1648636\ab/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_Output_layer_call_fn_1648625Oab/?,
%?"
 ?
inputs?????????
? "??????????8
__inference__creator_1648641?

? 
? "? 8
__inference__creator_1648659?

? 
? "? 8
__inference__creator_1648674?

? 
? "? 8
__inference__creator_1648692?

? 
? "? :
__inference__destroyer_1648654?

? 
? "? :
__inference__destroyer_1648669?

? 
? "? :
__inference__destroyer_1648687?

? 
? "? :
__inference__destroyer_1648702?

? 
? "? C
 __inference__initializer_1648649???

? 
? "? <
 __inference__initializer_1648664?

? 
? "? C
 __inference__initializer_1648682???

? 
? "? <
 __inference__initializer_1648697?

? 
? "? ?
"__inference__wrapped_model_1647354???????????YZab???
???
???
%?"
Temperature?????????
$?!

Luminosity?????????
 ?
Radius?????????
+?(
AbsoluteMagnitude?????????
#? 
	StarColor?????????
'?$
SpectralClass?????????
? "/?,
*
Output ?
Output?????????p
__inference_adapt_step_1648295N?C?@
9?6
4?1?
??????????IteratorSpec 
? "
 p
__inference_adapt_step_1648309N ?C?@
9?6
4?1?
??????????IteratorSpec 
? "
 p
__inference_adapt_step_1648356N)'(C?@
9?6
4?1?
??????????	IteratorSpec 
? "
 p
__inference_adapt_step_1648403N201C?@
9?6
4?1?
??????????IteratorSpec 
? "
 p
__inference_adapt_step_1648450N;9:C?@
9?6
4?1?
??????????IteratorSpec 
? "
 p
__inference_adapt_step_1648497NDBCC?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
Q__inference_category_encoding_20_layer_call_and_return_conditional_losses_1648536\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
6__inference_category_encoding_20_layer_call_fn_1648502O3?0
)?&
 ?
inputs?????????	

 
? "???????????
Q__inference_category_encoding_21_layer_call_and_return_conditional_losses_1648575\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
6__inference_category_encoding_21_layer_call_fn_1648541O3?0
)?&
 ?
inputs?????????	

 
? "???????????
E__inference_model_10_layer_call_and_return_conditional_losses_1647847???????????YZab???
???
???
%?"
Temperature?????????
$?!

Luminosity?????????
 ?
Radius?????????
+?(
AbsoluteMagnitude?????????
#? 
	StarColor?????????
'?$
SpectralClass?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_10_layer_call_and_return_conditional_losses_1647905???????????YZab???
???
???
%?"
Temperature?????????
$?!

Luminosity?????????
 ?
Radius?????????
+?(
AbsoluteMagnitude?????????
#? 
	StarColor?????????
'?$
SpectralClass?????????
p

 
? "%?"
?
0?????????
? ?
E__inference_model_10_layer_call_and_return_conditional_losses_1648116???????????YZab???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
p 

 
? "%?"
?
0?????????
? ?
E__inference_model_10_layer_call_and_return_conditional_losses_1648237???????????YZab???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
p

 
? "%?"
?
0?????????
? ?
*__inference_model_10_layer_call_fn_1647562???????????YZab???
???
???
%?"
Temperature?????????
$?!

Luminosity?????????
 ?
Radius?????????
+?(
AbsoluteMagnitude?????????
#? 
	StarColor?????????
'?$
SpectralClass?????????
p 

 
? "???????????
*__inference_model_10_layer_call_fn_1647789???????????YZab???
???
???
%?"
Temperature?????????
$?!

Luminosity?????????
 ?
Radius?????????
+?(
AbsoluteMagnitude?????????
#? 
	StarColor?????????
'?$
SpectralClass?????????
p

 
? "???????????
*__inference_model_10_layer_call_fn_1647953???????????YZab???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
p 

 
? "???????????
*__inference_model_10_layer_call_fn_1647995???????????YZab???
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
p

 
? "??????????{
__inference_restore_fn_1648729YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? {
__inference_restore_fn_1648756Y K?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_1648721?&?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_1648748? &?#
?
?
checkpoint_key 
? "???
`?]

name?
0/name 
#

slice_spec?
0/slice_spec 

tensor?
0/tensor
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
%__inference_signature_wrapper_1648281???????????YZab???
? 
???
@
AbsoluteMagnitude+?(
AbsoluteMagnitude?????????
2

Luminosity$?!

Luminosity?????????
*
Radius ?
Radius?????????
8
SpectralClass'?$
SpectralClass?????????
0
	StarColor#? 
	StarColor?????????
4
Temperature%?"
Temperature?????????"/?,
*
Output ?
Output?????????