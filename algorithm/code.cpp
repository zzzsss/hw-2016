#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <string.h>
#include <vector>
#include <math.h>
#define PB push_back
#define PDD pair<double,double>
using namespace std;

struct robot
{
	// robot with positin (x,y) and movement dx
	robot (double xx,double yy) {x=xx;y=yy;dx=0.;}
	double x,y,dx;
};
bool x_order (const robot& a,const robot& b) 
{
	return a.x<b.x;
}

struct overlap
{
	// overlap [l,r] caused by robots lidx, ridx
	overlap (double ll,double rr, int li, int ri) {l=ll;r=rr;lidx=li;ridx=ri;size=rr-ll;}
	double l,r,size;
	int lidx,ridx;
};

PDD right_shift(double radius,overlap o, PDD g, vector<robot>& robots)
{
	// return size, cost
	int tot= 0,count = 0;
	double mn;
	for(int i =o.ridx;i<robots.size() &&robots[i].x+radius<=g.first;i++){
		tot++;
		if(robots[i].dx<0.){
			count++;
			if(count==1) {mn = -robots[i].dx;}
			else mn = min(-robots[i].dx,mn);
		}
	}
	double p = count>0?mn:o.size;
	double cost = tot-count-count;
	if(o.r<0.){
		cost-=o.r;
	}
	return PDD(p,cost);
}
PDD left_shift(double L, double radius,overlap o, PDD g, vector<robot>& robots)
{
	//return size, cost
	int tot = 0;
	for(int i =o.lidx;i>=0&&robots[i].x-radius>=g.second;i--){
		tot++;
	}
	double cost = tot*1.;	
	if(o.l>L){
		cost+=o.l-L;
	}
	return PDD(o.size,cost);
}
double weak_single(double L,double radius,vector<robot>& robots)
{
	// weak one-layer BID: with L-length border and radius; 
	// return minimum cost and the corressponding optimal arrangement is recorderd
	
	sort(robots.begin(),robots.end(),x_order);
	// find all overlaps
	vector<overlap> overlaps;
	int n = robots.size();
	for(int i =0;i<n;i++){
		if(robots[i].x-radius<0.){
			overlaps.PB(overlap(robots[i].x-radius,min(0.,robots[i].x+radius),i,i));
		}
		if(robots[i].x+radius>L){
			overlaps.PB(overlap(max(L,robots[i].x-radius),robots[i].x+radius,i,i));
		}
	}
	for(int i = 0;i<n-1;i++){
		double l = robots[i+1].x-radius;
		double r = robots[i].x+radius;
		if(l<r){
			overlaps.PB(overlap(l,r,i,i+1));
		}
	}
	// find all gaps
	vector<PDD> gaps;
	int leftmost,rightmost;
	for(leftmost =0;leftmost<n;leftmost++)
		if(robots[leftmost].x+radius>0.) break;
	for(rightmost =n-1;rightmost>=0;rightmost--)
		if(robots[rightmost].x-radius<L) break;
	if (leftmost>=n || rightmost <0){
		gaps.PB(PDD(0.,L));
	}else{
		if(robots[leftmost].x-radius>0.)
			gaps.PB(PDD(0,robots[leftmost].x-radius));
		for(int i = leftmost;i<rightmost;i++){
			if (robots[i].x+radius<robots[i+1].x-radius)
			gaps.PB(PDD(robots[i].x+radius,robots[i+1].x-radius));
		}
		if(robots[rightmost].x+radius<L){
			gaps.PB(PDD(robots[rightmost].x+radius,L));
		}
	}
	// do elimintae gaps
	int left_closet,right_closet,m = overlaps.size();
	PDD left_result,right_result;
	for(PDD g : gaps){
		double gap_size = g.second - g.first;
		while(gap_size>0.){
			// find the left closet and the right closet overlaps
			left_closet = right_closet = -1;
			for(int i =0;i<m;i++){
				if(overlaps[i].size <=0.) continue;
				if ( overlaps[i].r <= g.first && (left_closet<0 || overlaps[i].r>overlaps[left_closet].r ))
					left_closet = i;
				if ( overlaps[i].l >= g.second && (right_closet<0 || overlaps[i].l<overlaps[right_closet].l ))
					right_closet = i;
			}
			// evaluate the cost for using either overlap
			if (left_closet >=0)
				left_result = right_shift(radius,overlaps[left_closet],g,robots);
			if (right_closet >=0)
				right_result = left_shift(L,radius,overlaps[right_closet],g,robots);
			// just do it
			if(left_closet>=0 && (right_closet<0 || left_result.second<right_result.second)){
				double c = min(left_result.first,gap_size);
				gap_size-=c;
				overlaps[left_closet].size-=c;
				for(int i =overlaps[left_closet].ridx;i<robots.size() &&robots[i].x+radius<=g.first;i++){
					robots[i].dx+=c;
				}
			}else{
				double c = min(right_result.first,gap_size);
				gap_size-=c;
				overlaps[right_closet].size-=c;
				for(int i =overlaps[right_closet].lidx;i>=0&&robots[i].x-radius>=g.second;i--){
					robots[i].dx-=c;
				}
			}
		}
	}
	// calulate overall cost
	double cost = 0.;
	for(robot& x : robots){
		cost+=fabs(x.dx);
	}
	return cost;
}
int main()
{
	freopen("in", "r", stdin);
	//freopen("out", "w", stdout);
	
	// all information we can get
	double L,radius,x,y;
	vector<robot> robots;
	scanf("%lf %lf",&L,&radius);
	while(scanf("%lf %lf",&x,&y)!=EOF){
		robots.PB(robot(x,y));
	}

	// call weak_single to do the task
	double cost = weak_single(L,radius,robots);
	
	//answer
	for(auto &z : robots){
		printf("%f ", z.x+z.dx);
	}
	return 0;
}