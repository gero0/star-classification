??
??
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68??	
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1548852*
value_dtype0	
?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1548763*
value_dtype0	
q
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1548989*
value_dtype0	
?
MutableHashTable_1MutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1548900*
value_dtype0	
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
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
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
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
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
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R 
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R 
?
Const_4Const*
_output_shapes
:*
dtype0*?
value?B?BREDBBLUEB
BLUE-WHITEBWHITEBYELLOW-WHITEBYELLOWISH WHITEBWHITISHB	YELLOWISHBWHITE-YELLOWBPALE YELLOW ORANGEBORANGEBBLUE-WHITE 
?
Const_5Const*
_output_shapes
:*
dtype0	*u
valuelBj	"`                                                        	       
                     
c
Const_6Const*
_output_shapes
:*
dtype0*(
valueBBMBOBBBABFBKBG
?
Const_7Const*
_output_shapes
:*
dtype0	*M
valueDBB	"8                                                 
?
StatefulPartitionedCallStatefulPartitionedCall
hash_tableConst_4Const_5*
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
__inference_<lambda>_1620943
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
__inference_<lambda>_1620948
?
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_1Const_6Const_7*
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
__inference_<lambda>_1620956
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
__inference_<lambda>_1620961
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
?4
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
L
lookup_table
token_counts
	keras_api
_adapt_function*
L
lookup_table
token_counts
	keras_api
_adapt_function*
* 
* 
* 
* 
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses*
?

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate1m~2m9m?:m?1v?2v?9v?:v?*
 
12
23
94
:5*
 
10
21
92
:3*
* 
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Kserving_default* 
R
L_initializer
M_create_resource
N_initialize
O_destroy_resource* 
?
P_create_resource
Q_initialize
R_destroy_resource<
table3layer_with_weights-0/token_counts/.ATTRIBUTES/table*
* 
* 
R
S_initializer
T_create_resource
U_initialize
V_destroy_resource* 
?
W_create_resource
X_initialize
Y_destroy_resource<
table3layer_with_weights-1/token_counts/.ATTRIBUTES/table*
* 
* 
* 
* 
* 
?
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 
* 
* 
]W
VARIABLE_VALUEHidden/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEHidden/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

10
21*

10
21*
* 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEOutput/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEOutput/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

90
:1*

90
:1*
* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
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
* 
b
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
12*

s0
t1*
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
8
	utotal
	vcount
w	variables
x	keras_api*
H
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

u0
v1*

w	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

y0
z1*

|	variables*
?z
VARIABLE_VALUEAdam/Hidden/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Hidden/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Hidden/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Hidden/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/Output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/Output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
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
?
StatefulPartitionedCall_2StatefulPartitionedCall!serving_default_AbsoluteMagnitudeserving_default_Luminosityserving_default_Radiusserving_default_SpectralClassserving_default_StarColorserving_default_Temperaturehash_table_1Const
hash_tableConst_1Hidden/kernelHidden/biasOutput/kernelOutput/bias*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_1620648
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filename?MutableHashTable_lookup_table_export_values/LookupTableExportV2AMutableHashTable_lookup_table_export_values/LookupTableExportV2:1AMutableHashTable_1_lookup_table_export_values/LookupTableExportV2CMutableHashTable_1_lookup_table_export_values/LookupTableExportV2:1!Hidden/kernel/Read/ReadVariableOpHidden/bias/Read/ReadVariableOp!Output/kernel/Read/ReadVariableOpOutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp(Adam/Hidden/kernel/m/Read/ReadVariableOp&Adam/Hidden/bias/m/Read/ReadVariableOp(Adam/Output/kernel/m/Read/ReadVariableOp&Adam/Output/bias/m/Read/ReadVariableOp(Adam/Hidden/kernel/v/Read/ReadVariableOp&Adam/Hidden/bias/v/Read/ReadVariableOp(Adam/Output/kernel/v/Read/ReadVariableOp&Adam/Output/bias/v/Read/ReadVariableOpConst_8*&
Tin
2			*
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
 __inference__traced_save_1621076
?
StatefulPartitionedCall_4StatefulPartitionedCallsaver_filenameMutableHashTableMutableHashTable_1Hidden/kernelHidden/biasOutput/kernelOutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Hidden/kernel/mAdam/Hidden/bias/mAdam/Output/kernel/mAdam/Output/bias/mAdam/Hidden/kernel/vAdam/Hidden/bias/vAdam/Output/kernel/vAdam/Output/bias/v*#
Tin
2*
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
#__inference__traced_restore_1621155??
?

?
C__inference_Output_layer_call_and_return_conditional_losses_1620815

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
?
0
 __inference__initializer_1620876
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
?+
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620346
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	 
hidden_1620335:
hidden_1620337: 
output_1620340:
output_1620342:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_18/StatefulPartitionedCall?,category_encoding_19/StatefulPartitionedCall?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handlespectralclass<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handle	starcolor<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
,category_encoding_18/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_18/Identity:output:0*
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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060?
,category_encoding_19/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_19/Identity:output:0-^category_encoding_18/StatefulPartitionedCall*
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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096?
Input/PartitionedCallPartitionedCalltemperature
luminosityradiusabsolutemagnitude5category_encoding_18/StatefulPartitionedCall:output:05category_encoding_19/StatefulPartitionedCall:output:0*
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1620335hidden_1620337*
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1620340output_1620342*
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
C__inference_Output_layer_call_and_return_conditional_losses_1620139v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_18/StatefulPartitionedCall-^category_encoding_19/StatefulPartitionedCall/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_18/StatefulPartitionedCall,category_encoding_18/StatefulPartitionedCall2\
,category_encoding_19/StatefulPartitionedCall,category_encoding_19/StatefulPartitionedCall2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:T P
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
: 
?
?
__inference_adapt_step_1620676
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
?*
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620146

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	 
hidden_1620123:
hidden_1620125: 
output_1620140:
output_1620142:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_18/StatefulPartitionedCall?,category_encoding_19/StatefulPartitionedCall?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
,category_encoding_18/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_18/Identity:output:0*
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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060?
,category_encoding_19/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_19/Identity:output:0-^category_encoding_18/StatefulPartitionedCall*
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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096?
Input/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_35category_encoding_18/StatefulPartitionedCall:output:05category_encoding_19/StatefulPartitionedCall:output:0*
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1620123hidden_1620125*
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1620140output_1620142*
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
C__inference_Output_layer_call_and_return_conditional_losses_1620139v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_18/StatefulPartitionedCall-^category_encoding_19/StatefulPartitionedCall/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_18/StatefulPartitionedCall,category_encoding_18/StatefulPartitionedCall2\
,category_encoding_19/StatefulPartitionedCall,category_encoding_19/StatefulPartitionedCall2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:O K
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
: 
?
0
 __inference__initializer_1620843
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
?*
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620271

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	 
hidden_1620260:
hidden_1620262: 
output_1620265:
output_1620267:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_18/StatefulPartitionedCall?,category_encoding_19/StatefulPartitionedCall?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
,category_encoding_18/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_18/Identity:output:0*
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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060?
,category_encoding_19/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_19/Identity:output:0-^category_encoding_18/StatefulPartitionedCall*
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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096?
Input/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_35category_encoding_18/StatefulPartitionedCall:output:05category_encoding_19/StatefulPartitionedCall:output:0*
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1620260hidden_1620262*
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1620265output_1620267*
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
C__inference_Output_layer_call_and_return_conditional_losses_1620139v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_18/StatefulPartitionedCall-^category_encoding_19/StatefulPartitionedCall/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_18/StatefulPartitionedCall,category_encoding_18/StatefulPartitionedCall2\
,category_encoding_19/StatefulPartitionedCall,category_encoding_19/StatefulPartitionedCall2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:O K
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
: 
?
<
__inference__creator_1620853
identity??
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1548989*
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
?
?
%__inference_signature_wrapper_1620648
absolutemagnitude

luminosity

radius
spectralclass
	starcolor
temperature
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_1620001o
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 22
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
: 
?r
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620527
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	7
%hidden_matmul_readvariableop_resource:4
&hidden_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Hidden/BiasAdd/ReadVariableOp?Hidden/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?"category_encoding_18/Assert/Assert?"category_encoding_19/Assert/Assert?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????k
category_encoding_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_18/MaxMax"string_lookup_18/Identity:output:0#category_encoding_18/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_18/MinMin"string_lookup_18/Identity:output:0%category_encoding_18/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_18/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_18/CastCast$category_encoding_18/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_18/GreaterGreatercategory_encoding_18/Cast:y:0!category_encoding_18/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_18/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_18/Cast_1Cast&category_encoding_18/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_18/GreaterEqualGreaterEqual!category_encoding_18/Min:output:0category_encoding_18/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_18/LogicalAnd
LogicalAnd category_encoding_18/Greater:z:0%category_encoding_18/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_18/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
)category_encoding_18/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
"category_encoding_18/Assert/AssertAssert#category_encoding_18/LogicalAnd:z:02category_encoding_18/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
#category_encoding_18/bincount/ShapeShape"string_lookup_18/Identity:output:0#^category_encoding_18/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_18/bincount/ConstConst#^category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_18/bincount/ProdProd,category_encoding_18/bincount/Shape:output:0,category_encoding_18/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_18/bincount/Greater/yConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_18/bincount/GreaterGreater+category_encoding_18/bincount/Prod:output:00category_encoding_18/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_18/bincount/CastCast)category_encoding_18/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_18/bincount/Const_1Const#^category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_18/bincount/MaxMax"string_lookup_18/Identity:output:0.category_encoding_18/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_18/bincount/add/yConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_18/bincount/addAddV2*category_encoding_18/bincount/Max:output:0,category_encoding_18/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_18/bincount/mulMul&category_encoding_18/bincount/Cast:y:0%category_encoding_18/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_18/bincount/minlengthConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_18/bincount/MaximumMaximum0category_encoding_18/bincount/minlength:output:0%category_encoding_18/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_18/bincount/maxlengthConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_18/bincount/MinimumMinimum0category_encoding_18/bincount/maxlength:output:0)category_encoding_18/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_18/bincount/Const_2Const#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_18/bincount/DenseBincountDenseBincount"string_lookup_18/Identity:output:0)category_encoding_18/bincount/Minimum:z:0.category_encoding_18/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(k
category_encoding_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_19/MaxMax"string_lookup_19/Identity:output:0#category_encoding_19/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_19/MinMin"string_lookup_19/Identity:output:0%category_encoding_19/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_19/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_19/CastCast$category_encoding_19/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_19/GreaterGreatercategory_encoding_19/Cast:y:0!category_encoding_19/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_19/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_19/Cast_1Cast&category_encoding_19/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_19/GreaterEqualGreaterEqual!category_encoding_19/Min:output:0category_encoding_19/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_19/LogicalAnd
LogicalAnd category_encoding_19/Greater:z:0%category_encoding_19/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_19/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
)category_encoding_19/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
"category_encoding_19/Assert/AssertAssert#category_encoding_19/LogicalAnd:z:02category_encoding_19/Assert/Assert/data_0:output:0#^category_encoding_18/Assert/Assert*

T
2*
_output_shapes
 ?
#category_encoding_19/bincount/ShapeShape"string_lookup_19/Identity:output:0#^category_encoding_19/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_19/bincount/ConstConst#^category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_19/bincount/ProdProd,category_encoding_19/bincount/Shape:output:0,category_encoding_19/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_19/bincount/Greater/yConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_19/bincount/GreaterGreater+category_encoding_19/bincount/Prod:output:00category_encoding_19/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_19/bincount/CastCast)category_encoding_19/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_19/bincount/Const_1Const#^category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_19/bincount/MaxMax"string_lookup_19/Identity:output:0.category_encoding_19/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_19/bincount/add/yConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_19/bincount/addAddV2*category_encoding_19/bincount/Max:output:0,category_encoding_19/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_19/bincount/mulMul&category_encoding_19/bincount/Cast:y:0%category_encoding_19/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_19/bincount/minlengthConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_19/bincount/MaximumMaximum0category_encoding_19/bincount/minlength:output:0%category_encoding_19/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_19/bincount/maxlengthConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_19/bincount/MinimumMinimum0category_encoding_19/bincount/maxlength:output:0)category_encoding_19/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_19/bincount/Const_2Const#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_19/bincount/DenseBincountDenseBincount"string_lookup_19/Identity:output:0)category_encoding_19/bincount/Minimum:z:0.category_encoding_19/bincount/Const_2:output:0*
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
Input/concatConcatV2inputs_0inputs_1inputs_2inputs_34category_encoding_18/bincount/DenseBincount:output:04category_encoding_19/bincount/DenseBincount:output:0Input/concat/axis:output:0*
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
NoOpNoOp^Hidden/BiasAdd/ReadVariableOp^Hidden/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp#^category_encoding_18/Assert/Assert#^category_encoding_19/Assert/Assert/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2>
Hidden/BiasAdd/ReadVariableOpHidden/BiasAdd/ReadVariableOp2<
Hidden/MatMul/ReadVariableOpHidden/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2H
"category_encoding_18/Assert/Assert"category_encoding_18/Assert/Assert2H
"category_encoding_19/Assert/Assert"category_encoding_19/Assert/Assert2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:Q M
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
: 
?
?
__inference_save_fn_1620900
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
?
H
__inference__creator_1620871
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1548900*
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
?
o
6__inference_category_encoding_18_layer_call_fn_1620681

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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060o
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
?
?
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620715

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
?
<
__inference__creator_1620820
identity??
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	1548852*
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
?8
?

 __inference__traced_save_1621076
file_prefixJ
Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2L
Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1	L
Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2N
Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1	,
(savev2_hidden_kernel_read_readvariableop*
&savev2_hidden_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop3
/savev2_adam_hidden_kernel_m_read_readvariableop1
-savev2_adam_hidden_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop3
/savev2_adam_hidden_kernel_v_read_readvariableop1
-savev2_adam_hidden_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const_8

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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Fsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2Hsavev2_mutablehashtable_lookup_table_export_values_lookuptableexportv2_1Hsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2Jsavev2_mutablehashtable_1_lookup_table_export_values_lookuptableexportv2_1(savev2_hidden_kernel_read_readvariableop&savev2_hidden_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop/savev2_adam_hidden_kernel_m_read_readvariableop-savev2_adam_hidden_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop/savev2_adam_hidden_kernel_v_read_readvariableop-savev2_adam_hidden_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const_8"/device:CPU:0*
_output_shapes
 *(
dtypes
2			?
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
?: ::::::::: : : : : : : : : ::::::::: 2(
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
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	
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
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?
?
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096

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
?
?
)__inference_model_9_layer_call_fn_1620165
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_1620146o
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 22
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
: 
?r
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620620
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	7
%hidden_matmul_readvariableop_resource:4
&hidden_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity??Hidden/BiasAdd/ReadVariableOp?Hidden/MatMul/ReadVariableOp?Output/BiasAdd/ReadVariableOp?Output/MatMul/ReadVariableOp?"category_encoding_18/Assert/Assert?"category_encoding_19/Assert/Assert?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handleinputs_5<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handleinputs_4<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????k
category_encoding_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_18/MaxMax"string_lookup_18/Identity:output:0#category_encoding_18/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_18/MinMin"string_lookup_18/Identity:output:0%category_encoding_18/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_18/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_18/CastCast$category_encoding_18/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_18/GreaterGreatercategory_encoding_18/Cast:y:0!category_encoding_18/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_18/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_18/Cast_1Cast&category_encoding_18/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_18/GreaterEqualGreaterEqual!category_encoding_18/Min:output:0category_encoding_18/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_18/LogicalAnd
LogicalAnd category_encoding_18/Greater:z:0%category_encoding_18/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_18/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
)category_encoding_18/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
"category_encoding_18/Assert/AssertAssert#category_encoding_18/LogicalAnd:z:02category_encoding_18/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
#category_encoding_18/bincount/ShapeShape"string_lookup_18/Identity:output:0#^category_encoding_18/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_18/bincount/ConstConst#^category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_18/bincount/ProdProd,category_encoding_18/bincount/Shape:output:0,category_encoding_18/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_18/bincount/Greater/yConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_18/bincount/GreaterGreater+category_encoding_18/bincount/Prod:output:00category_encoding_18/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_18/bincount/CastCast)category_encoding_18/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_18/bincount/Const_1Const#^category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_18/bincount/MaxMax"string_lookup_18/Identity:output:0.category_encoding_18/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_18/bincount/add/yConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_18/bincount/addAddV2*category_encoding_18/bincount/Max:output:0,category_encoding_18/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_18/bincount/mulMul&category_encoding_18/bincount/Cast:y:0%category_encoding_18/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_18/bincount/minlengthConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_18/bincount/MaximumMaximum0category_encoding_18/bincount/minlength:output:0%category_encoding_18/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_18/bincount/maxlengthConst#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_18/bincount/MinimumMinimum0category_encoding_18/bincount/maxlength:output:0)category_encoding_18/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_18/bincount/Const_2Const#^category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_18/bincount/DenseBincountDenseBincount"string_lookup_18/Identity:output:0)category_encoding_18/bincount/Minimum:z:0.category_encoding_18/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(k
category_encoding_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_19/MaxMax"string_lookup_19/Identity:output:0#category_encoding_19/Const:output:0*
T0	*
_output_shapes
: m
category_encoding_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
category_encoding_19/MinMin"string_lookup_19/Identity:output:0%category_encoding_19/Const_1:output:0*
T0	*
_output_shapes
: ]
category_encoding_19/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :w
category_encoding_19/CastCast$category_encoding_19/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
category_encoding_19/GreaterGreatercategory_encoding_19/Cast:y:0!category_encoding_19/Max:output:0*
T0	*
_output_shapes
: _
category_encoding_19/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : {
category_encoding_19/Cast_1Cast&category_encoding_19/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
!category_encoding_19/GreaterEqualGreaterEqual!category_encoding_19/Min:output:0category_encoding_19/Cast_1:y:0*
T0	*
_output_shapes
: ?
category_encoding_19/LogicalAnd
LogicalAnd category_encoding_19/Greater:z:0%category_encoding_19/GreaterEqual:z:0*
_output_shapes
: ?
!category_encoding_19/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
)category_encoding_19/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
"category_encoding_19/Assert/AssertAssert#category_encoding_19/LogicalAnd:z:02category_encoding_19/Assert/Assert/data_0:output:0#^category_encoding_18/Assert/Assert*

T
2*
_output_shapes
 ?
#category_encoding_19/bincount/ShapeShape"string_lookup_19/Identity:output:0#^category_encoding_19/Assert/Assert*
T0	*
_output_shapes
:?
#category_encoding_19/bincount/ConstConst#^category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
"category_encoding_19/bincount/ProdProd,category_encoding_19/bincount/Shape:output:0,category_encoding_19/bincount/Const:output:0*
T0*
_output_shapes
: ?
'category_encoding_19/bincount/Greater/yConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
%category_encoding_19/bincount/GreaterGreater+category_encoding_19/bincount/Prod:output:00category_encoding_19/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
"category_encoding_19/bincount/CastCast)category_encoding_19/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
%category_encoding_19/bincount/Const_1Const#^category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
!category_encoding_19/bincount/MaxMax"string_lookup_19/Identity:output:0.category_encoding_19/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
#category_encoding_19/bincount/add/yConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
!category_encoding_19/bincount/addAddV2*category_encoding_19/bincount/Max:output:0,category_encoding_19/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
!category_encoding_19/bincount/mulMul&category_encoding_19/bincount/Cast:y:0%category_encoding_19/bincount/add:z:0*
T0	*
_output_shapes
: ?
'category_encoding_19/bincount/minlengthConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_19/bincount/MaximumMaximum0category_encoding_19/bincount/minlength:output:0%category_encoding_19/bincount/mul:z:0*
T0	*
_output_shapes
: ?
'category_encoding_19/bincount/maxlengthConst#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
%category_encoding_19/bincount/MinimumMinimum0category_encoding_19/bincount/maxlength:output:0)category_encoding_19/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
%category_encoding_19/bincount/Const_2Const#^category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
+category_encoding_19/bincount/DenseBincountDenseBincount"string_lookup_19/Identity:output:0)category_encoding_19/bincount/Minimum:z:0.category_encoding_19/bincount/Const_2:output:0*
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
Input/concatConcatV2inputs_0inputs_1inputs_2inputs_34category_encoding_18/bincount/DenseBincount:output:04category_encoding_19/bincount/DenseBincount:output:0Input/concat/axis:output:0*
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
NoOpNoOp^Hidden/BiasAdd/ReadVariableOp^Hidden/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp#^category_encoding_18/Assert/Assert#^category_encoding_19/Assert/Assert/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2>
Hidden/BiasAdd/ReadVariableOpHidden/BiasAdd/ReadVariableOp2<
Hidden/MatMul/ReadVariableOpHidden/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp2H
"category_encoding_18/Assert/Assert"category_encoding_18/Assert/Assert2H
"category_encoding_19/Assert/Assert"category_encoding_19/Assert/Assert2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:Q M
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
: 
?
o
6__inference_category_encoding_19_layer_call_fn_1620720

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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096o
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
?
?
__inference_adapt_step_1620662
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
?
.
__inference__destroyer_1620833
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
?
?
__inference_<lambda>_1620943:
6key_value_init1548851_lookuptableimportv2_table_handle2
.key_value_init1548851_lookuptableimportv2_keys4
0key_value_init1548851_lookuptableimportv2_values	
identity??)key_value_init1548851/LookupTableImportV2?
)key_value_init1548851/LookupTableImportV2LookupTableImportV26key_value_init1548851_lookuptableimportv2_table_handle.key_value_init1548851_lookuptableimportv2_keys0key_value_init1548851_lookuptableimportv2_values*	
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
NoOpNoOp*^key_value_init1548851/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1548851/LookupTableImportV2)key_value_init1548851/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
)__inference_model_9_layer_call_fn_1620408
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_1620146o
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 22
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
: 
?
?
)__inference_model_9_layer_call_fn_1620316
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalltemperature
luminosityradiusabsolutemagnitude	starcolorspectralclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_1620271o
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 22
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
: 
?	
?
B__inference_Input_layer_call_and_return_conditional_losses_1620775
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
?
?
__inference_restore_fn_1620908
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
.
__inference__destroyer_1620881
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
?
?
)__inference_model_9_layer_call_fn_1620434
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_9_layer_call_and_return_conditional_losses_1620271o
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
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 22
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
: 
?
.
__inference__destroyer_1620866
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
?
?
__inference_restore_fn_1620935
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
?

?
C__inference_Output_layer_call_and_return_conditional_losses_1620139

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
?
?
 __inference__initializer_1620861:
6key_value_init1548988_lookuptableimportv2_table_handle2
.key_value_init1548988_lookuptableimportv2_keys4
0key_value_init1548988_lookuptableimportv2_values	
identity??)key_value_init1548988/LookupTableImportV2?
)key_value_init1548988/LookupTableImportV2LookupTableImportV26key_value_init1548988_lookuptableimportv2_table_handle.key_value_init1548988_lookuptableimportv2_keys0key_value_init1548988_lookuptableimportv2_values*	
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
NoOpNoOp*^key_value_init1548988/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1548988/LookupTableImportV2)key_value_init1548988/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620754

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
?
?
__inference_save_fn_1620927
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
?a
?
#__inference__traced_restore_1621155
file_prefixM
Cmutablehashtable_table_restore_lookuptableimportv2_mutablehashtable: Q
Gmutablehashtable_table_restore_1_lookuptableimportv2_mutablehashtable_1: 0
assignvariableop_hidden_kernel:,
assignvariableop_1_hidden_bias:2
 assignvariableop_2_output_kernel:,
assignvariableop_3_output_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: :
(assignvariableop_13_adam_hidden_kernel_m:4
&assignvariableop_14_adam_hidden_bias_m::
(assignvariableop_15_adam_output_kernel_m:4
&assignvariableop_16_adam_output_bias_m::
(assignvariableop_17_adam_hidden_kernel_v:4
&assignvariableop_18_adam_hidden_bias_v::
(assignvariableop_19_adam_output_kernel_v:4
&assignvariableop_20_adam_output_bias_v:
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?2MutableHashTable_table_restore/LookupTableImportV2?4MutableHashTable_table_restore_1/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B8layer_with_weights-0/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-0/token_counts/.ATTRIBUTES/table-valuesB8layer_with_weights-1/token_counts/.ATTRIBUTES/table-keysB:layer_with_weights-1/token_counts/.ATTRIBUTES/table-valuesB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2			?
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
AssignVariableOpAssignVariableOpassignvariableop_hidden_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_hidden_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_output_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_output_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_5IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_6IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_7IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_8IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0^

Identity_9IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp(assignvariableop_13_adam_hidden_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_hidden_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_output_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_output_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_hidden_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_hidden_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_output_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_output_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_93^MutableHashTable_table_restore/LookupTableImportV25^MutableHashTable_table_restore_1/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*C
_input_shapes2
0: : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
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
?
 __inference__initializer_1620828:
6key_value_init1548851_lookuptableimportv2_table_handle2
.key_value_init1548851_lookuptableimportv2_keys4
0key_value_init1548851_lookuptableimportv2_values	
identity??)key_value_init1548851/LookupTableImportV2?
)key_value_init1548851/LookupTableImportV2LookupTableImportV26key_value_init1548851_lookuptableimportv2_table_handle.key_value_init1548851_lookuptableimportv2_keys0key_value_init1548851_lookuptableimportv2_values*	
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
NoOpNoOp*^key_value_init1548851/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1548851/LookupTableImportV2)key_value_init1548851/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?+
?
D__inference_model_9_layer_call_and_return_conditional_losses_1620376
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclass?
;string_lookup_19_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_19_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_18_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_18_none_lookup_lookuptablefindv2_default_value	 
hidden_1620365:
hidden_1620367: 
output_1620370:
output_1620372:
identity??Hidden/StatefulPartitionedCall?Output/StatefulPartitionedCall?,category_encoding_18/StatefulPartitionedCall?,category_encoding_19/StatefulPartitionedCall?.string_lookup_18/None_Lookup/LookupTableFindV2?.string_lookup_19/None_Lookup/LookupTableFindV2?
.string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_19_none_lookup_lookuptablefindv2_table_handlespectralclass<string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_19/IdentityIdentity7string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
.string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_18_none_lookup_lookuptablefindv2_table_handle	starcolor<string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
string_lookup_18/IdentityIdentity7string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
,category_encoding_18/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_18/Identity:output:0*
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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060?
,category_encoding_19/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_19/Identity:output:0-^category_encoding_18/StatefulPartitionedCall*
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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620096?
Input/PartitionedCallPartitionedCalltemperature
luminosityradiusabsolutemagnitude5category_encoding_18/StatefulPartitionedCall:output:05category_encoding_19/StatefulPartitionedCall:output:0*
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109?
Hidden/StatefulPartitionedCallStatefulPartitionedCallInput/PartitionedCall:output:0hidden_1620365hidden_1620367*
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122?
Output/StatefulPartitionedCallStatefulPartitionedCall'Hidden/StatefulPartitionedCall:output:0output_1620370output_1620372*
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
C__inference_Output_layer_call_and_return_conditional_losses_1620139v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Hidden/StatefulPartitionedCall^Output/StatefulPartitionedCall-^category_encoding_18/StatefulPartitionedCall-^category_encoding_19/StatefulPartitionedCall/^string_lookup_18/None_Lookup/LookupTableFindV2/^string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2@
Hidden/StatefulPartitionedCallHidden/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall2\
,category_encoding_18/StatefulPartitionedCall,category_encoding_18/StatefulPartitionedCall2\
,category_encoding_19/StatefulPartitionedCall,category_encoding_19/StatefulPartitionedCall2`
.string_lookup_18/None_Lookup/LookupTableFindV2.string_lookup_18/None_Lookup/LookupTableFindV22`
.string_lookup_19/None_Lookup/LookupTableFindV2.string_lookup_19/None_Lookup/LookupTableFindV2:T P
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
: 
?
?
(__inference_Hidden_layer_call_fn_1620784

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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122o
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
?
?
(__inference_Output_layer_call_fn_1620804

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
C__inference_Output_layer_call_and_return_conditional_losses_1620139o
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
??
?
"__inference__wrapped_model_1620001
temperature

luminosity

radius
absolutemagnitude
	starcolor
spectralclassG
Cmodel_9_string_lookup_19_none_lookup_lookuptablefindv2_table_handleH
Dmodel_9_string_lookup_19_none_lookup_lookuptablefindv2_default_value	G
Cmodel_9_string_lookup_18_none_lookup_lookuptablefindv2_table_handleH
Dmodel_9_string_lookup_18_none_lookup_lookuptablefindv2_default_value	?
-model_9_hidden_matmul_readvariableop_resource:<
.model_9_hidden_biasadd_readvariableop_resource:?
-model_9_output_matmul_readvariableop_resource:<
.model_9_output_biasadd_readvariableop_resource:
identity??%model_9/Hidden/BiasAdd/ReadVariableOp?$model_9/Hidden/MatMul/ReadVariableOp?%model_9/Output/BiasAdd/ReadVariableOp?$model_9/Output/MatMul/ReadVariableOp?*model_9/category_encoding_18/Assert/Assert?*model_9/category_encoding_19/Assert/Assert?6model_9/string_lookup_18/None_Lookup/LookupTableFindV2?6model_9/string_lookup_19/None_Lookup/LookupTableFindV2?
6model_9/string_lookup_19/None_Lookup/LookupTableFindV2LookupTableFindV2Cmodel_9_string_lookup_19_none_lookup_lookuptablefindv2_table_handlespectralclassDmodel_9_string_lookup_19_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
!model_9/string_lookup_19/IdentityIdentity?model_9/string_lookup_19/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:??????????
6model_9/string_lookup_18/None_Lookup/LookupTableFindV2LookupTableFindV2Cmodel_9_string_lookup_18_none_lookup_lookuptablefindv2_table_handle	starcolorDmodel_9_string_lookup_18_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:??????????
!model_9/string_lookup_18/IdentityIdentity?model_9/string_lookup_18/None_Lookup/LookupTableFindV2:values:0*
T0	*'
_output_shapes
:?????????s
"model_9/category_encoding_18/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 model_9/category_encoding_18/MaxMax*model_9/string_lookup_18/Identity:output:0+model_9/category_encoding_18/Const:output:0*
T0	*
_output_shapes
: u
$model_9/category_encoding_18/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
 model_9/category_encoding_18/MinMin*model_9/string_lookup_18/Identity:output:0-model_9/category_encoding_18/Const_1:output:0*
T0	*
_output_shapes
: e
#model_9/category_encoding_18/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
!model_9/category_encoding_18/CastCast,model_9/category_encoding_18/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
$model_9/category_encoding_18/GreaterGreater%model_9/category_encoding_18/Cast:y:0)model_9/category_encoding_18/Max:output:0*
T0	*
_output_shapes
: g
%model_9/category_encoding_18/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_9/category_encoding_18/Cast_1Cast.model_9/category_encoding_18/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)model_9/category_encoding_18/GreaterEqualGreaterEqual)model_9/category_encoding_18/Min:output:0'model_9/category_encoding_18/Cast_1:y:0*
T0	*
_output_shapes
: ?
'model_9/category_encoding_18/LogicalAnd
LogicalAnd(model_9/category_encoding_18/Greater:z:0-model_9/category_encoding_18/GreaterEqual:z:0*
_output_shapes
: ?
)model_9/category_encoding_18/Assert/ConstConst*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
1model_9/category_encoding_18/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*^
valueUBS BMInput values must be in the range 0 <= values < num_tokens with num_tokens=13?
*model_9/category_encoding_18/Assert/AssertAssert+model_9/category_encoding_18/LogicalAnd:z:0:model_9/category_encoding_18/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 ?
+model_9/category_encoding_18/bincount/ShapeShape*model_9/string_lookup_18/Identity:output:0+^model_9/category_encoding_18/Assert/Assert*
T0	*
_output_shapes
:?
+model_9/category_encoding_18/bincount/ConstConst+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
*model_9/category_encoding_18/bincount/ProdProd4model_9/category_encoding_18/bincount/Shape:output:04model_9/category_encoding_18/bincount/Const:output:0*
T0*
_output_shapes
: ?
/model_9/category_encoding_18/bincount/Greater/yConst+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
-model_9/category_encoding_18/bincount/GreaterGreater3model_9/category_encoding_18/bincount/Prod:output:08model_9/category_encoding_18/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
*model_9/category_encoding_18/bincount/CastCast1model_9/category_encoding_18/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
-model_9/category_encoding_18/bincount/Const_1Const+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
)model_9/category_encoding_18/bincount/MaxMax*model_9/string_lookup_18/Identity:output:06model_9/category_encoding_18/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
+model_9/category_encoding_18/bincount/add/yConst+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
)model_9/category_encoding_18/bincount/addAddV22model_9/category_encoding_18/bincount/Max:output:04model_9/category_encoding_18/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
)model_9/category_encoding_18/bincount/mulMul.model_9/category_encoding_18/bincount/Cast:y:0-model_9/category_encoding_18/bincount/add:z:0*
T0	*
_output_shapes
: ?
/model_9/category_encoding_18/bincount/minlengthConst+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
-model_9/category_encoding_18/bincount/MaximumMaximum8model_9/category_encoding_18/bincount/minlength:output:0-model_9/category_encoding_18/bincount/mul:z:0*
T0	*
_output_shapes
: ?
/model_9/category_encoding_18/bincount/maxlengthConst+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
-model_9/category_encoding_18/bincount/MinimumMinimum8model_9/category_encoding_18/bincount/maxlength:output:01model_9/category_encoding_18/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
-model_9/category_encoding_18/bincount/Const_2Const+^model_9/category_encoding_18/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
3model_9/category_encoding_18/bincount/DenseBincountDenseBincount*model_9/string_lookup_18/Identity:output:01model_9/category_encoding_18/bincount/Minimum:z:06model_9/category_encoding_18/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(s
"model_9/category_encoding_19/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ?
 model_9/category_encoding_19/MaxMax*model_9/string_lookup_19/Identity:output:0+model_9/category_encoding_19/Const:output:0*
T0	*
_output_shapes
: u
$model_9/category_encoding_19/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       ?
 model_9/category_encoding_19/MinMin*model_9/string_lookup_19/Identity:output:0-model_9/category_encoding_19/Const_1:output:0*
T0	*
_output_shapes
: e
#model_9/category_encoding_19/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :?
!model_9/category_encoding_19/CastCast,model_9/category_encoding_19/Cast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
$model_9/category_encoding_19/GreaterGreater%model_9/category_encoding_19/Cast:y:0)model_9/category_encoding_19/Max:output:0*
T0	*
_output_shapes
: g
%model_9/category_encoding_19/Cast_1/xConst*
_output_shapes
: *
dtype0*
value	B : ?
#model_9/category_encoding_19/Cast_1Cast.model_9/category_encoding_19/Cast_1/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: ?
)model_9/category_encoding_19/GreaterEqualGreaterEqual)model_9/category_encoding_19/Min:output:0'model_9/category_encoding_19/Cast_1:y:0*
T0	*
_output_shapes
: ?
'model_9/category_encoding_19/LogicalAnd
LogicalAnd(model_9/category_encoding_19/Greater:z:0-model_9/category_encoding_19/GreaterEqual:z:0*
_output_shapes
: ?
)model_9/category_encoding_19/Assert/ConstConst*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
1model_9/category_encoding_19/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*]
valueTBR BLInput values must be in the range 0 <= values < num_tokens with num_tokens=8?
*model_9/category_encoding_19/Assert/AssertAssert+model_9/category_encoding_19/LogicalAnd:z:0:model_9/category_encoding_19/Assert/Assert/data_0:output:0+^model_9/category_encoding_18/Assert/Assert*

T
2*
_output_shapes
 ?
+model_9/category_encoding_19/bincount/ShapeShape*model_9/string_lookup_19/Identity:output:0+^model_9/category_encoding_19/Assert/Assert*
T0	*
_output_shapes
:?
+model_9/category_encoding_19/bincount/ConstConst+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB: ?
*model_9/category_encoding_19/bincount/ProdProd4model_9/category_encoding_19/bincount/Shape:output:04model_9/category_encoding_19/bincount/Const:output:0*
T0*
_output_shapes
: ?
/model_9/category_encoding_19/bincount/Greater/yConst+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
value	B : ?
-model_9/category_encoding_19/bincount/GreaterGreater3model_9/category_encoding_19/bincount/Prod:output:08model_9/category_encoding_19/bincount/Greater/y:output:0*
T0*
_output_shapes
: ?
*model_9/category_encoding_19/bincount/CastCast1model_9/category_encoding_19/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: ?
-model_9/category_encoding_19/bincount/Const_1Const+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"       ?
)model_9/category_encoding_19/bincount/MaxMax*model_9/string_lookup_19/Identity:output:06model_9/category_encoding_19/bincount/Const_1:output:0*
T0	*
_output_shapes
: ?
+model_9/category_encoding_19/bincount/add/yConst+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
)model_9/category_encoding_19/bincount/addAddV22model_9/category_encoding_19/bincount/Max:output:04model_9/category_encoding_19/bincount/add/y:output:0*
T0	*
_output_shapes
: ?
)model_9/category_encoding_19/bincount/mulMul.model_9/category_encoding_19/bincount/Cast:y:0-model_9/category_encoding_19/bincount/add:z:0*
T0	*
_output_shapes
: ?
/model_9/category_encoding_19/bincount/minlengthConst+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
-model_9/category_encoding_19/bincount/MaximumMaximum8model_9/category_encoding_19/bincount/minlength:output:0-model_9/category_encoding_19/bincount/mul:z:0*
T0	*
_output_shapes
: ?
/model_9/category_encoding_19/bincount/maxlengthConst+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0	*
value	B	 R?
-model_9/category_encoding_19/bincount/MinimumMinimum8model_9/category_encoding_19/bincount/maxlength:output:01model_9/category_encoding_19/bincount/Maximum:z:0*
T0	*
_output_shapes
: ?
-model_9/category_encoding_19/bincount/Const_2Const+^model_9/category_encoding_19/Assert/Assert*
_output_shapes
: *
dtype0*
valueB ?
3model_9/category_encoding_19/bincount/DenseBincountDenseBincount*model_9/string_lookup_19/Identity:output:01model_9/category_encoding_19/bincount/Minimum:z:06model_9/category_encoding_19/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output([
model_9/Input/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_9/Input/concatConcatV2temperature
luminosityradiusabsolutemagnitude<model_9/category_encoding_18/bincount/DenseBincount:output:0<model_9/category_encoding_19/bincount/DenseBincount:output:0"model_9/Input/concat/axis:output:0*
N*
T0*'
_output_shapes
:??????????
$model_9/Hidden/MatMul/ReadVariableOpReadVariableOp-model_9_hidden_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_9/Hidden/MatMulMatMulmodel_9/Input/concat:output:0,model_9/Hidden/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%model_9/Hidden/BiasAdd/ReadVariableOpReadVariableOp.model_9_hidden_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_9/Hidden/BiasAddBiasAddmodel_9/Hidden/MatMul:product:0-model_9/Hidden/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
model_9/Hidden/ReluRelumodel_9/Hidden/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
$model_9/Output/MatMul/ReadVariableOpReadVariableOp-model_9_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
model_9/Output/MatMulMatMul!model_9/Hidden/Relu:activations:0,model_9/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%model_9/Output/BiasAdd/ReadVariableOpReadVariableOp.model_9_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_9/Output/BiasAddBiasAddmodel_9/Output/MatMul:product:0-model_9/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
model_9/Output/SoftmaxSoftmaxmodel_9/Output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????o
IdentityIdentity model_9/Output/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp&^model_9/Hidden/BiasAdd/ReadVariableOp%^model_9/Hidden/MatMul/ReadVariableOp&^model_9/Output/BiasAdd/ReadVariableOp%^model_9/Output/MatMul/ReadVariableOp+^model_9/category_encoding_18/Assert/Assert+^model_9/category_encoding_19/Assert/Assert7^model_9/string_lookup_18/None_Lookup/LookupTableFindV27^model_9/string_lookup_19/None_Lookup/LookupTableFindV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????: : : : : : : : 2N
%model_9/Hidden/BiasAdd/ReadVariableOp%model_9/Hidden/BiasAdd/ReadVariableOp2L
$model_9/Hidden/MatMul/ReadVariableOp$model_9/Hidden/MatMul/ReadVariableOp2N
%model_9/Output/BiasAdd/ReadVariableOp%model_9/Output/BiasAdd/ReadVariableOp2L
$model_9/Output/MatMul/ReadVariableOp$model_9/Output/MatMul/ReadVariableOp2X
*model_9/category_encoding_18/Assert/Assert*model_9/category_encoding_18/Assert/Assert2X
*model_9/category_encoding_19/Assert/Assert*model_9/category_encoding_19/Assert/Assert2p
6model_9/string_lookup_18/None_Lookup/LookupTableFindV26model_9/string_lookup_18/None_Lookup/LookupTableFindV22p
6model_9/string_lookup_19/None_Lookup/LookupTableFindV26model_9/string_lookup_19/None_Lookup/LookupTableFindV2:T P
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
: 
?
,
__inference_<lambda>_1620948
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
?
,
__inference_<lambda>_1620961
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109

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
?

?
C__inference_Hidden_layer_call_and_return_conditional_losses_1620122

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
__inference__destroyer_1620848
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620795

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
?
H
__inference__creator_1620838
identity: ??MutableHashTable?
MutableHashTableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_nametable_1548763*
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
?
?
__inference_<lambda>_1620956:
6key_value_init1548988_lookuptableimportv2_table_handle2
.key_value_init1548988_lookuptableimportv2_keys4
0key_value_init1548988_lookuptableimportv2_values	
identity??)key_value_init1548988/LookupTableImportV2?
)key_value_init1548988/LookupTableImportV2LookupTableImportV26key_value_init1548988_lookuptableimportv2_table_handle.key_value_init1548988_lookuptableimportv2_keys0key_value_init1548988_lookuptableimportv2_values*	
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
NoOpNoOp*^key_value_init1548988/LookupTableImportV2*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init1548988/LookupTableImportV2)key_value_init1548988/LookupTableImportV2: 

_output_shapes
:: 

_output_shapes
:
?
?
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620060

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
?

?
'__inference_Input_layer_call_fn_1620764
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
B__inference_Input_layer_call_and_return_conditional_losses_1620109`
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
inputs/5"?L
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
StatefulPartitionedCall_2:0?????????tensorflow/serving/predict:ج
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-2
layer-11
layer_with_weights-3
layer-12
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
a
lookup_table
token_counts
	keras_api
_adapt_function"
_tf_keras_layer
a
lookup_table
token_counts
	keras_api
_adapt_function"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
?
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
?

9kernel
:bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Aiter

Bbeta_1

Cbeta_2
	Ddecay
Elearning_rate1m~2m9m?:m?1v?2v?9v?:v?"
	optimizer
<
12
23
94
:5"
trackable_list_wrapper
<
10
21
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_model_9_layer_call_fn_1620165
)__inference_model_9_layer_call_fn_1620408
)__inference_model_9_layer_call_fn_1620434
)__inference_model_9_layer_call_fn_1620316?
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
D__inference_model_9_layer_call_and_return_conditional_losses_1620527
D__inference_model_9_layer_call_and_return_conditional_losses_1620620
D__inference_model_9_layer_call_and_return_conditional_losses_1620346
D__inference_model_9_layer_call_and_return_conditional_losses_1620376?
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
"__inference__wrapped_model_1620001Temperature
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
Kserving_default"
signature_map
j
L_initializer
M_create_resource
N_initialize
O_destroy_resourceR jCustom.StaticHashTable
Q
P_create_resource
Q_initialize
R_destroy_resourceR Z
table??
"
_generic_user_object
?2?
__inference_adapt_step_1620662?
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
S_initializer
T_create_resource
U_initialize
V_destroy_resourceR jCustom.StaticHashTable
Q
W_create_resource
X_initialize
Y_destroy_resourceR Z
table??
"
_generic_user_object
?2?
__inference_adapt_step_1620676?
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
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_category_encoding_18_layer_call_fn_1620681?
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
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620715?
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
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
?2?
6__inference_category_encoding_19_layer_call_fn_1620720?
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
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620754?
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
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?2?
'__inference_Input_layer_call_fn_1620764?
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
B__inference_Input_layer_call_and_return_conditional_losses_1620775?
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
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Hidden_layer_call_fn_1620784?
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
C__inference_Hidden_layer_call_and_return_conditional_losses_1620795?
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
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?2?
(__inference_Output_layer_call_fn_1620804?
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
C__inference_Output_layer_call_and_return_conditional_losses_1620815?
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
 "
trackable_list_wrapper
~
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
12"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_1620648AbsoluteMagnitude
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
__inference__creator_1620820?
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
 __inference__initializer_1620828?
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
__inference__destroyer_1620833?
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
__inference__creator_1620838?
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
 __inference__initializer_1620843?
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
__inference__destroyer_1620848?
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
__inference__creator_1620853?
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
 __inference__initializer_1620861?
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
__inference__destroyer_1620866?
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
__inference__creator_1620871?
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
 __inference__initializer_1620876?
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
__inference__destroyer_1620881?
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
N
	utotal
	vcount
w	variables
x	keras_api"
_tf_keras_metric
^
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
u0
v1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
|	variables"
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
__inference_save_fn_1620900checkpoint_key"?
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
__inference_restore_fn_1620908restored_tensors_0restored_tensors_1"?
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
__inference_save_fn_1620927checkpoint_key"?
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
__inference_restore_fn_1620935restored_tensors_0restored_tensors_1"?
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
Const_7?
C__inference_Hidden_layer_call_and_return_conditional_losses_1620795\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_Hidden_layer_call_fn_1620784O12/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_Input_layer_call_and_return_conditional_losses_1620775????
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
'__inference_Input_layer_call_fn_1620764????
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
C__inference_Output_layer_call_and_return_conditional_losses_1620815\9:/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? {
(__inference_Output_layer_call_fn_1620804O9:/?,
%?"
 ?
inputs?????????
? "??????????8
__inference__creator_1620820?

? 
? "? 8
__inference__creator_1620838?

? 
? "? 8
__inference__creator_1620853?

? 
? "? 8
__inference__creator_1620871?

? 
? "? :
__inference__destroyer_1620833?

? 
? "? :
__inference__destroyer_1620848?

? 
? "? :
__inference__destroyer_1620866?

? 
? "? :
__inference__destroyer_1620881?

? 
? "? C
 __inference__initializer_1620828???

? 
? "? <
 __inference__initializer_1620843?

? 
? "? C
 __inference__initializer_1620861???

? 
? "? <
 __inference__initializer_1620876?

? 
? "? ?
"__inference__wrapped_model_1620001?
??129:???
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
__inference_adapt_step_1620662N?C?@
9?6
4?1?
??????????IteratorSpec 
? "
 p
__inference_adapt_step_1620676N?C?@
9?6
4?1?
??????????IteratorSpec 
? "
 ?
Q__inference_category_encoding_18_layer_call_and_return_conditional_losses_1620715\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
6__inference_category_encoding_18_layer_call_fn_1620681O3?0
)?&
 ?
inputs?????????	

 
? "???????????
Q__inference_category_encoding_19_layer_call_and_return_conditional_losses_1620754\3?0
)?&
 ?
inputs?????????	

 
? "%?"
?
0?????????
? ?
6__inference_category_encoding_19_layer_call_fn_1620720O3?0
)?&
 ?
inputs?????????	

 
? "???????????
D__inference_model_9_layer_call_and_return_conditional_losses_1620346?
??129:???
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
D__inference_model_9_layer_call_and_return_conditional_losses_1620376?
??129:???
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
? ?
D__inference_model_9_layer_call_and_return_conditional_losses_1620527?
??129:???
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
? ?
D__inference_model_9_layer_call_and_return_conditional_losses_1620620?
??129:???
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
)__inference_model_9_layer_call_fn_1620165?
??129:???
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
)__inference_model_9_layer_call_fn_1620316?
??129:???
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
)__inference_model_9_layer_call_fn_1620408?
??129:???
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
)__inference_model_9_layer_call_fn_1620434?
??129:???
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
__inference_restore_fn_1620908YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? {
__inference_restore_fn_1620935YK?H
A?>
?
restored_tensors_0
?
restored_tensors_1	
? "? ?
__inference_save_fn_1620900?&?#
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
__inference_save_fn_1620927?&?#
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
%__inference_signature_wrapper_1620648?
??129:???
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