module compare (input [3:0] a, input [3:0] b,
	output eq);
	wire [3:0] c; // bus definition
	
	// MyXnor i0 (.A(a[0]), .B(b[0]), .Z(c[0]) );
	// MyXnor i1 (.A(a[1]), .B(b[1]), .Z(c[1]) );
	// MyXnor i2 (.A(a[2]), .B(b[2]), .Z(c[2]) );
	// MyXnor i3 (.A(a[3]), .B(b[3]), .Z(c[3]) );
	
	assign c = ~(a ^ b); // XNOR
	assign eq = &c; // short format

endmodule
