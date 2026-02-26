module mux4(input [3:0] d0, d1, d2, d3,
	input [1:0] s,
	output [3:0] y);

	assign y = (s == 2’b11) ? d3 :
	(s == 2’b10) ? d2 :
	(s == 2’b01) ? d1 :
	d0;
	// if	(s = “11” ) then y= d3
	// else if (s = “10” ) then y= d2
	// else if (s = “01” ) then y= d1
	// else y= d0		

endmodule
