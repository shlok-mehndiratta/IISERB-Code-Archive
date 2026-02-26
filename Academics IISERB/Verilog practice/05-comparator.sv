module compare (input a0, a1, a2, a3, b0, b1, b2, b3,
	output eq);
	wire c0, c1, c2, c3, c01, c23;

	MyXnor i0 (.A(a0), .B(b0), .Z(c0) ); // XNOR
	MyXnor i1 (.A(a1), .B(b1), .Z(c1) ); // XNOR
	MyXnor i2 (.A(a2), .B(b2), .Z(c2) ); // XNOR
	MyXnor i3 (.A(a3), .B(b3), .Z(c3) ); // XNOR
	MyAnd h1 (.A(c0), .B(c1), .Z(c01) ); // AND
	MyAnd h2 (.A(c2), .B(c3), .Z(c23) ); // AND
	MyAnd h3 (.A(c01), .B(c23), .Z(eq) ); // AND


endmodule
