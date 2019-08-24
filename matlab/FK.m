function [ee,J1,J2]=FK(q0,q1,q2)
l1=[1;0];
J1=rz(q0)*l1;
J2=J1+rz(q0+q1)*l1;
ee=J2+rz(q0+q1+q2)*l1;
end