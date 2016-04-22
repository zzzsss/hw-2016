package training;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class LibsvmCommon extends Mach{
	public static int MODE_CLASS=0,MODE_REG=1,MODE_LINEAR=2;
	static Random RAND_N = new Random();
	static int DEF_NAME_LEN = 10;
	//filenames
	static String FNAME_SCALE = ".scale";
	static String FNAME_MODEL = ".model";
	static String FNAME_RANGE = ".range";
	static String FNAME_TRAIN = ".train";
	static String FNAME_TEST = ".test";
	static String FNAME_CVOUT = ".cvout";
	static String FNAME_OUT = ".output";
	//libsvm (on windows)
	static String LIBSVM_SCALE = "svm/libsvm/svm-scale.exe";
	static String LIBSVM_TRAIN = "svm/libsvm/svm-train.exe";
	static String LIBSVM_TEST = "svm/libsvm/svm-predict.exe";
	static String LIBSVM_GRID = "svm/test/grid.py";
	static int LIBSVM_CV_FOLD = 5;
	static String LIBSVM_OPTION_REG = " -s 3 ";
	//liblinear (for windows)
	static String LIBLINEAR_TRAIN = "svm/liblinear/train.exe";
	static String LIBLINEAR_TEST = "svm/liblinear/predict.exe";
	
	int mode;
	String svm_name;	//for the files of libsvm
	boolean need_scale = false;
	
	//-----------infomations---------------
	private void outputln_stdout(String x){
		System.out.println(x);
	}
	private void outputln_stderr(String x){
		System.err.println(x);
	}
	//------------
	private void deal_with_name(String name){
		if(name == null){
			StringBuilder s = new StringBuilder();
			for(int i=0;i<DEF_NAME_LEN;i++){
				s.append((char)('a'+RAND_N.nextInt(26)));
			}
			svm_name = s.toString();
		}
		else
			svm_name = name;
	}
	private void write_data(List<DataPoint> data,String fname){
		try{
			FileOutputStream out = new FileOutputStream(fname);
			PrintStream p = new PrintStream(out);
			for(DataPoint point : data){
				int[] t_index = point.get_index();
				double[] t_value = point.get_fvalue();
				p.print(point.get_value());
				p.print(" ");
				for(int jj=0;jj<t_index.length;jj++)
					p.print((1+t_index[jj])+":"+t_value[jj]+" ");
				p.println();
			}
			p.close();
			outputln_stdout("Finish Writing data-file: "+fname);
		}catch (FileNotFoundException e){
            e.printStackTrace();
        }
	}
	private void scale_data(String fname,String sname,String rangename,boolean store){
		try {
            Runtime run = Runtime.getRuntime();
            Process p = null;
            String cmd = "";
            if(store)
            	cmd = (LIBSVM_SCALE+" -l 0 -s "+rangename+" "+fname);
            else
            	cmd = (LIBSVM_SCALE+" -l 0 -r "+rangename+" "+fname);
            outputln_stdout("Executing: "+cmd);

            p = run.exec(cmd);
            BufferedInputStream in_out = new BufferedInputStream(p.getInputStream());
            BufferedReader inB_out = new BufferedReader(new InputStreamReader(in_out));
            BufferedInputStream in_err = new BufferedInputStream(p.getErrorStream());
            BufferedReader inB_err = new BufferedReader(new InputStreamReader(in_err));
            String lineStr=null;
            //redirection
            try{
    			FileOutputStream out = new FileOutputStream(sname);
    			PrintStream pr = new PrintStream(out);
                while ((lineStr = inB_out.readLine())!=null){
                    pr.println(lineStr);
                }
                pr.close();
    		}catch (FileNotFoundException e){
                e.printStackTrace();
            }
            while((lineStr = inB_err.readLine())!=null)
            	outputln_stderr(lineStr);
            p.waitFor();
		}catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
	}
	private String cross_v(String name,String cvout,String pass_options){
		try {
            Runtime run = Runtime.getRuntime();
            String cmd = "";
            if(mode != MODE_LINEAR){
            	cmd = ("python "+LIBSVM_GRID+" -gnuplot null -v "+LIBSVM_CV_FOLD
            			+" -svmtrain "+LIBSVM_TRAIN+" "+pass_options+" -out "+cvout+" "+name);
            }
            else{	//for liblinear
            	cmd = ("python "+LIBSVM_GRID+" -gnuplot null -v "+LIBSVM_CV_FOLD + " -log2c -14,14,1 -log2g 1,1,1 "
            			+" -svmtrain "+LIBLINEAR_TRAIN+" -out "+cvout+" "+name);
            }
            outputln_stdout("Executing: "+cmd);
            Process p = run.exec(cmd);
            BufferedInputStream in_out = new BufferedInputStream(p.getInputStream());
            BufferedReader inB_out = new BufferedReader(new InputStreamReader(in_out));
            BufferedInputStream in_err = new BufferedInputStream(p.getErrorStream());
            BufferedReader inB_err = new BufferedReader(new InputStreamReader(in_err));
            String lineStr=null,lineStr2=null;
            while (true){
                lineStr2 = lineStr;
                lineStr = inB_out.readLine();
                if(lineStr==null)
                	break;
                else
                	outputln_stdout(lineStr);
            }
            while((lineStr = inB_err.readLine())!=null)
            	outputln_stderr(lineStr);
            String [] result = lineStr2.split(" ");
            System.out.println("CV for "+svm_name+" "+lineStr2);
            if(mode != MODE_LINEAR)
            	return " -c "+result[0]+" -g "+result[1]+" ";
            else
            	return " -c "+result[0]+" ";
		}catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
	}
	private void libsvm_train(String fname,String mname,String pass){
		try {
            Runtime run = Runtime.getRuntime();
            String cmd = "";
            if(mode != MODE_LINEAR)
            	cmd = (LIBSVM_TRAIN+" "+pass+" "+fname+" "+mname);
            else
            	cmd = (LIBLINEAR_TRAIN+" "+pass+" "+fname+" "+mname);
            outputln_stdout("Executing: "+cmd);
            Process p = run.exec(cmd);
            BufferedInputStream in_out = new BufferedInputStream(p.getInputStream());
            BufferedReader inB_out = new BufferedReader(new InputStreamReader(in_out));
            BufferedInputStream in_err = new BufferedInputStream(p.getErrorStream());
            BufferedReader inB_err = new BufferedReader(new InputStreamReader(in_err));
            String lineStr=null;
            while ((lineStr = inB_out.readLine())!=null){
                outputln_stdout(lineStr);
            }
            while((lineStr = inB_err.readLine())!=null)
            	outputln_stderr(lineStr);
            p.waitFor();
		}catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
	}
	private void libsvm_test(String fname,String mname,String output){
		try {
            Runtime run = Runtime.getRuntime();
            String cmd = "";
            if(mode != MODE_LINEAR)
            	cmd = (LIBSVM_TEST+" "+fname+" "+mname+" "+output);
            else
            	cmd = (LIBLINEAR_TEST+" "+fname+" "+mname+" "+output);
            outputln_stdout("Executing: "+cmd);
            Process p = run.exec(cmd);
            BufferedInputStream in_out = new BufferedInputStream(p.getInputStream());
            BufferedReader inB_out = new BufferedReader(new InputStreamReader(in_out));
            BufferedInputStream in_err = new BufferedInputStream(p.getErrorStream());
            BufferedReader inB_err = new BufferedReader(new InputStreamReader(in_err));
            String lineStr=null;
            while ((lineStr = inB_out.readLine())!=null){
                outputln_stdout(lineStr);
            }
            while((lineStr = inB_err.readLine())!=null)
            	outputln_stderr(lineStr);
            p.waitFor();
		}catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
	}
	
	//--------------
	public LibsvmCommon(File filename){
		super(1);
		try{
			FileInputStream in = new FileInputStream(filename);
			Scanner sin = new Scanner(in);
			String name_of_mach="";
			if(!(name_of_mach=sin.next()).equals(NAMES[mach_id])){
				sin.close();
				throw new RuntimeException("Wrong mach file, should be svm, but "+name_of_mach);
			}
			svm_name = sin.next();
			mode = sin.nextInt();
			need_scale = sin.nextBoolean();
			sin.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
            throw new RuntimeException("File not exist: "+filename);
		}
	}
	
	public LibsvmCommon(int m,boolean scale){
		super(1);
		mode = m;
		need_scale = scale;
	}
	public LibsvmCommon(){	//default liblinear
		super(1);
		mode = MODE_LINEAR;
	}
	
	//--------
	public void train(List<DataPoint> index,Object others){
		String svm_n = (String)others;	//the base filename for the libsvm files
		deal_with_name(svm_n);
		List<DataPoint> training_data = index;
		//write training file
		String train_name = svm_name+FNAME_TRAIN;
		String scale_name = train_name;
		if(need_scale)
			scale_name = scale_name +FNAME_SCALE;
		String range_name = svm_name+FNAME_RANGE;
		String model_name = svm_name+FNAME_MODEL;
		String cvout_name = svm_name+FNAME_CVOUT;
		String reg_option = (mode != MODE_REG)?" ":LIBSVM_OPTION_REG;
		write_data(training_data,train_name);
		if(need_scale)
			scale_data(train_name,scale_name,range_name,true);
		String pass = cross_v(scale_name,cvout_name,reg_option);
		libsvm_train(scale_name,model_name,pass+reg_option);
	}
	public void train(int v,List<DataPoint>[] index,Object others){
		train(DataPoint.combine(index),others);
	}
	
	public double evaluate(DataPoint x){
		//not support
		throw new RuntimeException("No support for single point");
	}
	public List<Double> evaluate(List<DataPoint> testing_data){
		List<Double> ret = new ArrayList<Double>();
		//write test file
		String test_name = svm_name+FNAME_TEST;
		String scale_name = test_name;
		if(need_scale)
			scale_name = scale_name +FNAME_SCALE;
		String range_name = svm_name+FNAME_RANGE;
		String model_name = svm_name+FNAME_MODEL;
		String output_name = test_name+FNAME_OUT;

		write_data(testing_data,test_name);
		if(need_scale)
			scale_data(test_name,scale_name,range_name,false);
		libsvm_test(scale_name,model_name,output_name);
		try{
			FileInputStream in = new FileInputStream(output_name);
			Scanner sin = new Scanner(in);
			for(int j=0;j<testing_data.size();j++){
				ret.add(sin.nextDouble());
			}
			sin.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
            throw new RuntimeException();
		}
		return ret;
	}
	public void write(String filename){
		try{
			FileOutputStream out = new FileOutputStream(filename);
			PrintStream p = new PrintStream(out);
			p.println(NAMES[mach_id]);
			p.println(svm_name);
			p.println(mode);
			p.println(need_scale);
			p.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
        }
	}
	
	//test
	public static void main(String[] x){
		//test it
		Mach test_svm = null;
		//train
		
		
		File train_f = new File("svm\\test\\a1a.txt");
		try{
			FileInputStream in = new FileInputStream(train_f);
			Scanner sin = new Scanner(in);
			String line = "";
			List<DataPoint> tr = new ArrayList<DataPoint>();
			while(sin.hasNextLine()){
				line = sin.nextLine();
				String[] tmp = line.split(" ");
				int[] d_ind = new int[tmp.length-1];
				double[] d_fv = new double[tmp.length-1];
				double v = Double.parseDouble(tmp[0]);
				for(int i = 1;i<tmp.length;i++){
					String[] tmp_split = tmp[i].split(":");
					d_ind[i-1] = Integer.parseInt(tmp_split[0]); 
					d_fv[i-1] = Double.parseDouble(tmp_split[1]);
				}
				tr.add(new DataPoint(v,d_ind,d_fv));
			}
			test_svm = new LibsvmCommon(2,false);
			test_svm.train(tr, "svm\\test\\abc");
			test_svm.write("svm\\test\\abc.mach");
			sin.close();
			
		}catch (FileNotFoundException e){
            e.printStackTrace();
            throw new RuntimeException();
		}
		
		
		/*
		File test_f = new File("svm\\test\\a1a.t");
		try{
			FileInputStream in = new FileInputStream(test_f);
			Scanner sin = new Scanner(in);
			String line = "";
			List<DataPoint> tr = new ArrayList<DataPoint>();
			while(sin.hasNextLine()){
				line = sin.nextLine();
				String[] tmp = line.split(" ");
				int[] d_ind = new int[tmp.length-1];
				double[] d_fv = new double[tmp.length-1];
				double v = Double.parseDouble(tmp[0]);
				for(int i = 1;i<tmp.length;i++){
					String[] tmp_split = tmp[i].split(":");
					d_ind[i-1] = Integer.parseInt(tmp_split[0]); 
					d_fv[i-1] = Double.parseDouble(tmp_split[1]);
				}
				tr.add(new DataPoint(v,d_ind,d_fv));
			}
			test_svm = new LibsvmCommon(new File("testing/train"));
			List<Double> out = test_svm.evaluate(tr);
			System.out.println(out);
			sin.close();
		}catch (FileNotFoundException e){
            e.printStackTrace();
            throw new RuntimeException();
		}
		*/
		
	}
}

