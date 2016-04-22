package training;

import java.util.*;
import java.io.*;

public class NaiveBayes extends Mach{
	//multinomial naive bayes with simple add-one smoothing
	int classes;	//from 0 to ...
	int v_size;		//number of words
	double log_pcw[][];	//logs[C][V]
	double log_pc[];		//pc[C]
	
	public NaiveBayes(File filename){
		super(0);
		try{
			FileInputStream in = new FileInputStream(filename);
			Scanner sin = new Scanner(in);
			String name_of_mach="";
			if(!(name_of_mach=sin.next()).equals(NAMES[mach_id])){
				sin.close();
				throw new RuntimeException("Wrong mach file, should be svm, but "+name_of_mach);
			}
			classes = sin.nextInt();
			v_size = sin.nextInt();
			log_pcw = new double[classes][v_size];
			log_pc = new double[classes];
			for(int i=0;i<classes;i++){
				log_pc[i] = sin.nextDouble();
			}
			for(int i=0;i<classes;i++)
				for(int j=0;j<v_size;j++)
					log_pcw[i][j] = sin.nextDouble();
			sin.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
            throw new RuntimeException("File not exist: "+filename);
		}
	}
	public NaiveBayes(){
		super(0);
	}
	
	public void train(List<DataPoint> index,Object others){
		//not support
		throw new RuntimeException("No support for this mode.");
	}
	
	public void train(int v,List<DataPoint>[] index,Object others){
		//index should be of size c
		classes = index.length;
		v_size = v;
		log_pcw = new double[classes][v_size];
		log_pc = new double[classes];
		int pc_normalize = 0;
		for(int i=0;i<classes;i++){
			List<DataPoint> t = index[i];
			log_pc[i] = t.size();	//temperary for counts
			pc_normalize += log_pc[i];
			//count for log_pcw[i]
			int pcw_count = v_size;
			for(int j=0;j<v_size;j++)
				log_pcw[i][j] = 1;	//smoothing
			for(DataPoint item : t){
				int [] ind = item.get_index();
				double [] fv = item.get_fvalue();
				for(int one=0;one<ind.length;one++){
					int num = (int)(fv[one]);
					pcw_count += num;
					log_pcw[i][ind[one]] += num;
				}
			}
			//normalize and log
			for(int n=0;n<v_size;n++)
				log_pcw[i][n] = Math.log(log_pcw[i][n]/pcw_count);
		}
		for(int i=0;i<classes;i++){
			log_pc[i] = Math.log(log_pc[i]/pc_normalize);
		}
	}
	
	public double evaluate(DataPoint x){
		//no check here --- here f can be null
		double []values = new double[classes];
		int [] ind = x.get_index();
		double [] fv = x.get_fvalue();
		for(int i=0;i<classes;i++){
			values[i] = log_pc[i];
			for(int tmp = 0; tmp < ind.length ; tmp++)
				values[i] += fv[tmp]*log_pcw[i][ind[tmp]];
		}
		int max_item = 0;
		double max_value = values[0];
		for(int i=1;i<classes;i++){
			if(values[i] > max_value){
				max_value = values[i];
				max_item = i;
			}
		}
		return max_item;
	}
	
	public List<Double> evaluate(List<DataPoint> x){
		List<Double> ret = new ArrayList<Double>();
		for(DataPoint index : x){
			ret.add(evaluate(index));
		}
		return ret;
	}
	
	public void write(String filename){
		try{
			FileOutputStream out = new FileOutputStream(filename);
			PrintStream p = new PrintStream(out);
			p.println(NAMES[mach_id]);
			p.printf("%d %d\n",classes,v_size);
			for(int i=0;i<classes;i++){
				p.printf("%f ",log_pc[i]);
				p.println();
			}
			for(int i=0;i<classes;i++){
				for(double one : log_pcw[i])
					p.printf("%f ",one);
				p.println();
			}
			p.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
        }
	}
}
