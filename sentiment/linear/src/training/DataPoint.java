package training;

import java.util.*;
public class DataPoint {
	double value;
	int[] index;
	double[] fvalue;	//feature value --- default all 1
	
	//for the DataPoint --- ind and fv must be same size
	public DataPoint(double v,int[] ind,double[] fv){
		value = v;
		index = ind;
		fvalue = fv;
	}
	public DataPoint(double v,int[] ind){
		value = v;
		index = ind;
		fvalue = new double[ind.length];
		Arrays.fill(fvalue, 1.0);
	}
	public double get_value(){
		return value;
	}
	public void set_value(double v){
		value = v;
	}
	public int[] get_index(){
		return index;
	}
	public double[] get_fvalue(){
		return fvalue;
	}
	
	//for convinient of classification
	public static List<DataPoint> combine(List<DataPoint>[] l){
		ArrayList<DataPoint> x = new ArrayList<DataPoint>();
		for(List<DataPoint> ll : l)
			x.addAll(ll);
		return x;
	}
}
