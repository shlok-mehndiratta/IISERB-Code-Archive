module top (A, SEL, B, Y);
	input A, SEL, Y;
	output Y;
	wire n1; 


// instantiate small once
small i_first (.A(A), .B(SEL), .Y(n1) );
small i_second (.A(n1), .B(C), .Y(Y));


endmodule

//Alternativley we can write in short form - 
// small i_first (A, SEL, n1);
//small i_second (A, B, Y);
