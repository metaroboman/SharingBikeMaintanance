function z = Boundary(Vdelta,h2,S2,t)
z = max(0,(Vdelta/(2*t))*(h2*S2/(Vdelta^2)-t));
end