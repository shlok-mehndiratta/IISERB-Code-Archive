;logical AND, OR and XOR operations
;output displayed using 7447 IC

.include "/home/shlok-mehndiratta/m328Pdef.inc"

ldi r16, 0b00111100 ;identifying output pins 2,3,4,5
out DDRD,r16		;declaring pins as output



ldi r16,0b00000001	;initializing W
ldi r17,0b00000001	;initializing X
ldi r18,0b00000000	;initializing Y
ldi r19,0b00000000      ;initializing Z

mov r1,r16
mov r2,r17
mov r3,r18
mov r4,r19


; defining r24 for complement operations

; A = W' (Keeping r5 fixed for A)
mov r24,r16
rcall comp
mov r5,r24


;B = WX'Z'+W'X  (Keeping r6 fixed for B)
mov r6,r1

mov r24,r2
rcall comp

and r6,r24

mov r24,r4
rcall comp

and r6,r24

mov r24,r1
rcall comp

and r24,r2

or r6,r24


;C = WXY'+X'Y+W'Y (Keeping r7 fixed for C)
mov r7,r1
and  r7,r2
mov r24,r3
rcall comp
and r7,r24

mov r24,r2
rcall comp
and r24,r3

or r7,r24

mov r24,r1
rcall comp
and r24,r3

or r7,r24


; D = WXY+W'Z (Keeping r8 fixed for D)
mov r8,r1
and r8,r2
and r8,r3

mov r24,r1
rcall comp

and r24,r4
or r8,r24


;following code is for displaying output
;shifting LSB to 1st, 2nd & 3rd  position
ldi r20, 0b00000010	;counter = 2
ldi r21, 0b00000011	;counter = 3
ldi r22, 0b00000100	;counter = 4
ldi r23, 0b00000101	;counter = 5


rcall loopw2
rcall loopw3
rcall loopw4
rcall loopw5	;calling the loopw routine

or r5,r6
or r5,r7
or r5,r8


out PORTD,r5		;writing output to pins 2,3,4,5



Start:
rjmp Start

;loop for bit shifting
loopw2: lsl r5
	dec r20
	brne	loopw2
	ret

loopw3: lsl r6
	dec r21
	brne	loopw3
	ret

loopw4:	lsl r7			;left shift
	dec r22			;counter --
	brne	loopw4		;if counter != 0
	ret

loopw5: lsl r8
	dec r23
	brne	loopw5
	ret

comp:
	mov r0,r24
	ldi r24,0b00000001
	eor r24,r0
	ret
