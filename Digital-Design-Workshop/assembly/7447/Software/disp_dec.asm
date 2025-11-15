;Display Decoder 
;output displayed using 7447 IC

.include "/home/shlok-mehndiratta/m328Pdef.inc"

;This sets pin 0, 1, 2, 3, 4, 5, 6, 7 as output pins
ldi r16, 0b01111111
out DDRD, r16

;Defining A, B, C, D
ldi r21, 0b00000001 ;A 
ldi r22, 0b00000000 ;B
ldi r23, 0b00000001 ;C
ldi r24, 0b00000000 ;D

;Logic for ABCDEFG

;a = CB'A'+D'C'B'A (Keeping r1 fixed for a)
mov r1,r23
mov r20,r22
rcall comp
and r1,r20
mov r20,r21
rcall comp
and r1,r20

mov r20,r24
rcall comp
mov r25,r21
and r25,r20
mov r20,r23
rcall comp
and r25,r20
mov r20,r22
rcall comp
and r25,r20

or r1,r25

; b = CB'A+CBA' (Keeping r2 for b)
mov r2,r23
mov r25,r21
eor r25,r22
and r2,r25

;c = C'BA' (Keeping r3 for c)
mov r3,r22
mov r20,r21
rcall comp
and r3,r20
mov r20,r23
rcall comp
and r3,r20

;d = B'(C'A+CA')+CBA (Keeping r4 for d)
mov r20,r22
rcall comp
mov r4,r20
mov r25,r21
eor r25,r23
and r4,r25
mov r25,r21
and r25,r22
and r25,r23

or r4,r25


;e = A + CB' (r5 for e)
mov r5,r21
mov r25,r23
mov r20,r22
rcall comp
and r25,r20
or r5,r25


;f = D'C'A+BA+C'B (r6 for f)
mov r20,r24
rcall comp
mov r6,r20
mov r20,r23
rcall comp
and r6,r20
and r6,r21
mov r25,r22
and r25,r21
or r6,r25

mov r25,r22
mov r20,r23
rcall comp
and r25,r20
or r6,r25

;g = D'C'B'+CBA (r7 for g)
mov r7, r21
and r7,r22
and r7,r23
mov r20,r22
rcall comp
mov r25,r20
mov r20,r23
rcall comp
and r25,r20
mov r20,r24
rcall comp
and r25,r20
or r7,r25

; Defining counters

ldi r16,0b00000001
ldi r17,0b00000010
ldi r18,0b00000011
ldi r19,0b00000100
ldi r20,0b00000101
ldi r25,0b00000110

rcall loopw1
rcall loopw2
rcall loopw3
rcall loopw4
rcall loopw5
rcall loopw6

mov r25,r1
or r25,r2
or r25,r3
or r25,r4
or r25,r5
or r25,r6
or r25,r7

out PORTD,r25


start:
	rjmp start

loopw1:	lsl r2
	dec r16
	brne	loopw1
	ret

loopw2:	lsl r3
	dec r17
	brne 	loopw2
	ret
loopw3:	lsl r4
	dec r18
	brne 	loopw3
	ret
loopw4:	lsl r5
	dec r19
	brne 	loopw4
	ret

loopw5:	lsl r6
	dec r20
	brne 	loopw5
	ret
loopw6: lsl r7
	dec r25
	brne 	loopw6
	ret

comp:	mov r0,r20
	ldi r20, 0b00000001
	eor r20,r0
	ret


