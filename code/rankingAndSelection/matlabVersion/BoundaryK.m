function z = Boundary(Vdelta,h2,S2,t)
z = max(0,h2*S2/t-Vdelta/2);
end