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

function checker(numpts, sqrsize)
%function produces checker board data for 2 class classification
%problem - code might generate unequal number of points in each class
% numpts is the total number of points to be generated
% sqrsize is the length of the square
% the generated points are written a file :checker.dat
ii = 1;
jj = 1;
fid = fopen('checker.dat','w');
for idx = 1:numpts
    x = rand * sqrsize;
    y = rand * sqrsize;
    pt = [ x y]';
    %there are just two types of indices -
    %one with sum even and other with sum odd
    lb = [floor(x) floor(y)]';
    %all indices with sum even belong to one color
    %sum of two even numbers and sum of two odd
    %numbers is always even - one color

    modlen = length(find(mod(lb,2)));
    %modlen = 0 if two indices are even
    %modlen = 2 if two indices are odd
        if (modlen == 0 || modlen == 2)
        pt = [-1 ; pt];
        out1(:,ii) = pt;
        ii = ii + 1;
        else
        pt = [1 ; pt];
        out2(:,jj) = pt;
        jj = jj + 1;
        end
    end
 plot(out1(2,:), out1(3,:),'.r', ...
    out2(2,:), out2(3,:),'+g');

   fprintf(fid, '%d %f %f\n', out1);
   fprintf(fid, '%d %f %f\n', out2);
   fclose(fid);
