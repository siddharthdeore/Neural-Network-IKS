function plt_fk(q0,q1,q2)
[ee,J1,J2]=FK(q0,q1,q2);
plot([0,J1(1),J2(1),ee(1)],[0,J1(2),J2(2),ee(2)])
axis([-3 3 -3 3])
end