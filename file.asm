;Miscrosoft Calling Convention
;x86-64 Calling Convention

includelib ucrt.lib
includelib legacy_stdio_definitions.lib

ExitProcess PROTO

EXTERN GetStdHandle: PROC
EXTERN WriteFile: PROC
EXTERN printf: PROC
EXTERN scanf: PROC

.data
	;Array for Storing NN
	NeuralNet dq 1, 2, 3, 4		;Arbitrary Values

	;Strings needed for output
	message db "Hello World", 0	;Testing Message
	startString db " Main Linear Function Starting at %3.4lf X + %3.4lf  |  Neural Network Starting at %3.4lf X + %3.4lf", 10
	MORSOMEGATOSI db " ", 0
	trainingString db "Beginning Training...\n", 0
	epochTrainingString db "Epoch %d: NN Weight %lf, Bias %lf\n", 0
	finalNetworkString db "Main Linear Function Starting at %3.4lf X + %3.4lf  |  Neural Network Finalizing at %3.4lf X + %3.4lf", 7
	MORSOMEGATOSII db " ", 0
	testingValueString db "Testing x = %d: Expected %lf, Got %lf\n", 0

	;HYPER denotes that the constant is a Hyperparameter of the NN
	HYPERALPHA dq 0.0001			;Alpha
	HYPEREPOCHS dq 10000000		;Epochs -- MUST BE LARGE PAST 1,000,000 FOR GOOD RESULTS

	;Define Floats to use with MOVSD because for some reason MASM hates Floats
	notquiteone dq 0.999	;Used in Adjusting the Perceptron
	oneFloat dq 1.0			
	twoFloat dq 2.0
	halfFloat dq 0.5
	minBoundFloat dq 0.01

.code

;Randomized the values of the Neural Net Array
RANDOMIZEARRAY PROC
	RDRAND RAX					;Create a random value between 1-128 and convert to a float
	AND RAX, 127
	ADD RAX, 1
	cvtsi2sd xmm0, RAX
	MOVSD [NeuralNet], xmm0
	RDRAND RAX
	AND RAX, 127
	ADD RAX, 1
	cvtsi2sd xmm0, RAX
	MOVSD [NeuralNet+1*8], xmm0
	RDRAND RAX
	AND RAX, 127
	ADD RAX, 1
	cvtsi2sd xmm0, RAX
	MOVSD [NeuralNet+2*8], xmm0
	RDRAND RAX
	AND RAX, 127
	ADD RAX, 1
	cvtsi2sd xmm0, RAX
	MOVSD [NeuralNet+3*8], xmm0
	RET
RANDOMIZEARRAY ENDP

;Calculates the value of a sigmoid function
;	sig = 1 / (1 + e^-z)
;Recieves the XMM0 register and performs a Sigmoid Calulation on it
;e^x is precise to a Taylor Series of 50 times
;Modifies values stored in xmm0, xmm1, xmm2, xmm3, rdx, r9
;Takes in a Float xmm0 (z)
;Outputs a Value that is stored in xmm0
SIGMOID PROC
	mov r9, -1			;Multiply z (xmm0) by -1
	cvtsi2sd xmm1, r9
	mulsd xmm0, xmm1
	mov rdx, 0			;Prerequisite for EXP
	sub rsp, 40
	call EXP			;e^-z
	add rsp, 40
	addsd xmm0, xmm3	;xmm3 should still be 1 after calling EXP
	divsd xmm3, xmm0	;1 / (1 + x^-z)
	movsd xmm0, xmm3	;Keep result in xmm0

	;NEED TO IMPURE THE RESULT, TO PERHAPS 4 PRECISION
	mov r12, 3F5h
	mov r13, 3FEh
	SUB rsp, 40
	CALL BOUNDSD
	ADD rsp, 40	

	ret
SIGMOID ENDP

;Restricts a Double to a set amount of accuracy
;	x*10^y <= INPUT VALUE <= x*10^z
;	Where y and z are the exponents bounds
;Modified values in xmm0, r12, r13, r14
;Takes in Values of xmm0 (INPUT VALUE), r12 (Lower Bound), r13 (Upper Bound)
;	Bounds are 3 byte values compareative to the Exponent of the Double (SD)
;Outputs a Bounded SD into xmm0
BOUNDSD PROC
	;Prepare Bounds for Comparing to the Exponent
	shl r12, 52
	shl r13, 52
	
	;Make r14 a Mask for extracting the Exponent
	MOV r14, 7FF0h
	shl r14, 48
	MOVQ r8, xmm0
	AND r14, r8	;Extract Exponent
	
	;Compare r14 to the Bounds
	;USE SWITCH STATEMENT ESQ IMPLEMENTATION
	cmp r14, r12
	jl TINYBABY
	jge BIGMAN

	TINYBABY:
		movq xmm0, minBoundFloat
		jmp ENDBOUNDSD

	BIGMAN:
		movq xmm0, oneFloat
		jmp ENDBOUNDSD

	ENDBOUNDSD:

	RET
BOUNDSD ENDP


;Recursive Function Used during the SIGMOID function to calculate exponentials
;	e^x = 1 + x/1 (1 + x/2 (1 + x/3 ( ... -> 1+ x/50)))
;modifies values stored in xmm0, xmm1, xmm2, xmm3 , rdx, r9
;RDX should be modified to 0 beforehand for maintaining accuracy
;Takes in a Value x that is stored in xmm0
;Outputs a Value that is stored in xmm0
EXP PROC
	MOVSD XMM2, XMM0		;xmm2 will hold x
	ADD rdx, 1				;Let rdx be an accuracy counter going to 50
	cvtsi2sd xmm1, rdx		;convert rdx to a float
	cmp rdx, 50				;Check if the Accuracy is up to the desired bound
	mov r9, 1				;1 is used in every recursision, so we need that
	jg FULLACCURACY
	jmp EXPRECURSIVE

	EXPRECURSIVE:			;Recursive Call the exponential to full accuracy of 1 + xmm0 / acc * (recursive)
		SUB rsp, 32
		call EXP
		ADD rsp, 32
		SUBSD XMM1, oneFloat	;Fried Egg on Focalors
		MULSD XMM0, XMM2		;recusion mul x
		divsd xmm0, xmm1		;div by count
		mov r9, 1
		cvtsi2sd xmm3, r9
		addsd xmm0, xmm3		;LOOP UNROLL START
		RET					;End the Function Call

	FULLACCURACY:			;End of the Recursive Calls
		divsd xmm0, xmm1	;Do the First Recursive of 1 + x/accuracy
		mov r9, 1
		cvtsi2sd xmm3, r9
		addsd xmm0, xmm3
		RET					;End the Function Call
EXP ENDP


;Adjusts the Weight and Bias of a Perceptron
;modifies values stored in xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, rcx, rdx, r9
;Takes in data at xmm0 (weight) and xmm1 (bias) and xmm2 (test value), xmm3 (z_real)
;Stores data in xmm0 (weight) and xmm1 (bias)
;Currently this is Hardcoded to adjust directly to the array, but with pointers could adjust Tensors
ADJUSTPERCEPTRON PROC
	movsd xmm4, xmm0			;Test weight into xmm4
	movsd xmm6, xmm2			;Test Value into xmm6
	movsd xmm5, xmm1			;Test Bias into xmm5
	movsd xmm12, xmm3			;z_real into xmm12
	sub rsp, 32
	CALL LINEARFUNCTION		;Store z_pred in xmm0
	add rsp, 32
	SUBSD xmm12, xmm0		;Get back Original Weight, do (z_real - z_pred)

	movsd xmm7, xmm12			;(z_real - z_pred) into xmm7

	sub rsp, 32
	call SIGMOID			;Sigmoid should be in xmm0
	add rsp, 32

	MULSD xmm7, xmm0		;(z_real - z_pred) * sigmoid(z_pred)
	;MULSD xmm7, twoFloat
	movsd xmm1, notquiteone	
	SUBSD xmm1, xmm0		
	
	MULSD xmm7, xmm1		;(z_real - z_pred) * sigmoid(z_pred) * (0.999 - sigmoid(z_pred))
	MULSD xmm7, HYPERALPHA	;alpha * (z_real - z_pred) * sigmoid(z_pred) * (0.999 - sigmoid(z_pred))

	;STORE xmm5 INTO THE ARRAY FOR TEST BIAS
	addsd xmm5, xmm7				;addsd works better than subsd??????
	MOVSD [NeuralNet + 3*8], xmm5
	movsd xmm11, xmm5
	
	
	MULSD xmm7, xmm6		;alpha * (z_real - z_pred) * sigmoid(z_pred) * (0.999 - sigmoid(z_pred)) * xi
	;STORE xmm4 INTO THE ARRAY FOR TEST WEIGHT
	addsd xmm4, xmm7
	;alpha * ((z_real - z_pred) * sigmoid(z_pred) * (0.999 - sigmoid(z_pred)) * xi)
	MOVSD [NeuralNet + 2*8], xmm4
	movsd xmm10, xmm4

	RET
ADJUSTPERCEPTRON ENDP


;Models the simple Linear function
;	z = mx + b
;Modifies value in xmm0
;Takes in xmm0 (weight), xmm1 (bias), xmm2 (test variable)
;Outputs value in xmm0 (z)
LINEARFUNCTION PROC
	MULSD xmm0, xmm2
	ADDSD xmm0, xmm1
	ret
LINEARFUNCTION ENDP


;Trains a Neural Network
;	for (epochs)  {(Neurons) -> (Better Neurons)}
;Modifies Values Placed in the Array (Below)
;Takes in an array of Two Networks (1 neuron each for this project, a fuller version would be dynamic)
;	[Weight True, Bias True, Weight Train, Bias Train]
;Outputs is the Referenced Array With Modified Train Values
TRAINNETWORK PROC
	;Print Inital NN
	;FOR TEST
	movsd xmm10, xmm0
	movsd xmm11, xmm1

	mov rcx, offset startString
	movsd xmm0, NeuralNet 
	movsd xmm1, NeuralNet +1*8
	movsd xmm2, NeuralNet +2*8
	movsd xmm3, NeuralNet +3*8
	sub rsp, 40
	;call printf
	add rsp, 40

	;BEGIN LOOP OVER EPOCHS
	mov r10, 0		;use r10 as the Epoch counter
	EPOCHTRAINLOOP:
		
		;Get a Random Value from 0 - 500 (domain) and apply that for the Training
		RDRAND RAX
		AND RAX, 370
		cvtsi2sd xmm2, RAX

		;Get the Expected Value	
		movsd xmm0, [NeuralNet]			;True Weight
		movsd xmm1, [NeuralNet + 1*8]	;True Bias
		sub rsp, 32
		call LINEARFUNCTION		;xmm0 should now by the z_real
		add rsp, 32

		;Train the Expectation against the Test NN
		movsd xmm3, xmm0				;xmm3 z_real
		movsd xmm0, [NeuralNet + 2*8]	;xmm0 Test Weight
		movsd xmm1, [NeuralNet + 3*8]	;xmm1 Test Bias
									;xmm2 Should still be X
		sub rsp, 32
		call ADJUSTPERCEPTRON
		add rsp, 32

		;If after 1000 iterations -> Print current Progress
		mov r11, r10
		AND r11, 1000
		cmp r11, 1000
		;jz PRINTITERATION

		;Test to see if neurons adjust
		movsd xmm2, xmm10
		movsd xmm3, xmm11
		jmp CONTINUEEPOCH

		PRINTITERATION:
			MOV RCX, offset epochTrainingString
			mov rax, r10
			movsd xmm0, [NeuralNet + 2*8]
			movsd xmm1, [NeuralNet + 3*8]
			sub rsp, 40
			;call printf
			add rsp, 40


		CONTINUEEPOCH:

		;Continue to train for Epochs
		INC r10
		cmp r10, HYPEREPOCHS
		jnz EPOCHTRAINLOOP

	;Print Final NN Weight and Bias
	mov rcx, offset finalNetworkString
	movsd xmm0, [NeuralNet] 
	movsd xmm1, [NeuralNet +1*8]
	movsd xmm2, [NeuralNet +2*8]
	movsd xmm3, [NeuralNet +3*8]

	sub rsp, 40
	;call printf
	add rsp, 40

	ret
TRAINNETWORK ENDP


;Prints out the initial Values of the Neural Netork
;Modifies Values in xmm0, xmm1, xmm2, xmm3
;Uses the values of NeuralNet
PRINTINITIALNN PROC
	mov RCX, offset startString
	MOVSD xmm0, NeuralNet+0*8
	MOVQ RDX, xmm0
	MOVSD xmm1, NeuralNet+1*8
	MOVQ R8, xmm1
	MOVSD xmm3, NeuralNet+3*8
	MOVQ R9, xmm3
	PUSH R9
	MOVSD xmm2, NeuralNet+2*8
	MOVQ R9, xmm2
	sub RSP, 32
	call printf
	add RSP, 32
	POP R9
	ret
PRINTINITIALNN ENDP


;Tests the Neural Network and Outputs to the Screen the Expected and Given Values
;Modifies no Values
;Takes in an Array of Two Neural Networks
;	[Weight True, Bias True, Weight Train, Bias Train]
PRINTFINALNN PROC
	mov RCX, offset finalNetworkString
	MOVSD xmm0, NeuralNet+0*8
	MOVQ RDX, xmm0
	MOVSD xmm1, NeuralNet+1*8
	MOVQ R8, xmm1
	MOVSD xmm3, NeuralNet+3*8
	MOVQ R9, xmm3
	PUSH R9
	MOVSD xmm2, NeuralNet+2*8
	MOVQ R9, xmm2
	sub RSP, 32
	call printf
	add RSP, 32
	POP R9
	ret
PRINTFINALNN ENDP


;MAIN FUNCTION START
;Creates a Neural Network of 1 Neuron that models the Linear Function y=mx+b
;It will assume random values and output the start, finish, and training of the neurons
;Though since this is a 1 neuron system, it may fit better to call this a Neural rather than a Neural Network...
mainCRTStartup PROC
	;FOR FINDING NN IN MEMORY
	mov r15, offset NeuralNet


	;randomize Array
	sub rsp, 40
	call RANDOMIZEARRAY
	add rsp, 40

	;Print values of the NN Before Training
	sub rsp, 40
	call PRINTINITIALNN
	add rsp, 40

	;FOR TEST
	movsd xmm2, [NeuralNet +2*8]
	movsd xmm3, [NeuralNet +3*8]

	;train neural network
	sub rsp, 40
	call TRAINNETWORK
	add rsp, 40

	;print some output, namely both functions and a test point
	sub rsp, 40
	call PRINTFINALNN
	add rsp, 40

	;MOVSD RAX, XMM0
	PUSH rcx
	call ExitProcess

mainCRTStartup ENDP

END