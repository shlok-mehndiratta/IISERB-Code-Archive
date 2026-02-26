‘timescale 1ns/1ps	//unit/precision

module simple (input a, output z1, z2);

assign #5 z1 = ~a; // inverted output after 5ns
	assign #9 z2 = a; // output after 9ns


endmodule
