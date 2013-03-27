%% This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.
%
%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%GNU General Public License for more details.
%
%You should have received a copy of the GNU General Public License
%along with this program. If not, see <http://www.gnu.org/license
%s/>.
%%


function quadplot(n, r, fname)
% n is the number of points to generate inside and outside a circle of radius
% r. fname is the output file name to write the generated points

theta=0:pi/16:2*pi;
x=r*cos(theta);%2 is the radius
y=r*sin(theta);
%shift the center to (r,r) from (0,0) to avoid circle cross negative axes
x=x+r;
y=y+r;
ii = 1;
%generate points inside and outside the circle of radius r
while ii ~= n+1
        test=[rand*2*r rand*2*r];%square of side 2r
        if (test(:,1)^2+test(:,2)^2-2*r*(test(:,1)+test(:,2))+r*r <= 0)
                grp1(ii,:)=test;
                ii = ii + 1;
        end
end
ii = 1;
while ii ~= n + 1
        test=[rand*2*r rand*2*r];%square of side 2r
        if (test(:,1)^2+test(:,2)^2-2*r*(test(:,1)+test(:,2))+r*r > 0)
                grp2(ii,:)=test;
                ii = ii + 1;
        end
end
grp1 = [ones(n,1) grp1];
grp2 = [-ones(n,1) grp2];
grp=[grp1;grp2];
plot(x,y,grp1(:,2),grp1(:,3),'*g',grp2(:,2),grp2(:,3),'*r');
grp=grp';
fid=fopen(fname,'w');
fprintf(fid,'%d %.4f %.4f\n',grp);
fclose(fid);
