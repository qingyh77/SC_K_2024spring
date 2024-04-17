function gamma = Circle_initpos(p1,p2,p3,r)
c = norm(p2-p3,'fro');
cos_gamma = (2*(r^2) - c^2)/( 2*r^2 );
if p3(2)<p1(2)
    gamma = 2*pi - acos(cos_gamma);
else
    gamma = acos(cos_gamma);
end

end