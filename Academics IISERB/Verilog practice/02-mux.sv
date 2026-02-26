module mux(input d0, d1, 
	input s, 
	output y);

	wire ns, y1, y2;

	not g1 (ns,s);
	and g2 (y1, d0, ns);
	and g3 (y3, d1, s);
	or g4 (y, y1, y2);

endmodule
