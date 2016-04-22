package text;

/* no need
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.trees.international.pennchinese.ChineseTreebankLanguagePack;
import edu.stanford.nlp.ling.CoreAnnotations;
*/
import nlpir.NLPIR.CLibrary;
import java.io.PrintStream;
import java.util.*;

public class Tools {
	//paths
	//1.nlpir
	static String nlpir_path = "lib";
	static String NO_POS = "0000";
	//1.5 tag
	static String pos_path = "models/chinese-distsim.tagger";
	//static MaxentTagger tagger;
	//2.ner
	static String ner_path = "models/chinese.misc.distsim.crf.ser.gz";
	//static AbstractSequenceClassifier<CoreLabel> ner_classifier;
	//3.parser
	static String parser_path = "models/chinesePCFG.ser.gz";
	//static LexicalizedParser parser;
	//-dependency
	//static TreebankLanguagePack tlp;
	//static GrammaticalStructureFactory gsf;
	
	//options
	static boolean ner_select = false;
	static boolean parser_select = false;
	
	public static boolean init(boolean ns,boolean ps){
		ner_select = ns;
		parser_select = ps;
		//1. nlpir: seg & pos
		int charset_type = 1;
		int init_flag = CLibrary.Instance.NLPIR_Init(nlpir_path, charset_type, "0");
		String nativeBytes = null;
		if (0 == init_flag) {
			nativeBytes = CLibrary.Instance.NLPIR_GetLastErrorMsg();
			System.err.println("初始化失败！fail reason is "+nativeBytes);
			return false;
		}
		/*
		//1.5
		tagger = new MaxentTagger(pos_path);
		//2. ner
		try{
			if(ner_select)
				ner_classifier = CRFClassifier.getClassifier(ner_path);
		}catch(Exception e){
			System.err.println("NER Classifier init failed...");
			return false;
		}
		//3.parser
		if(parser_select){
			parser = LexicalizedParser.loadModel(parser_path);
			tlp = new ChineseTreebankLanguagePack();
			gsf = tlp.grammaticalStructureFactory();
		}
		*/
		return true;
	}
	public static boolean init(){
		return init(true,true);
	}
	public static void deinit(){
		CLibrary.Instance.NLPIR_Exit();
		/*
		tagger = null;
		ner_classifier = null;
		parser = null;
		*/
	}
	
	//1.seg & pos
	/* segment example of stanford seg
	System.setOut(new PrintStream(System.out, true, "GBK"));
	Properties props = new Properties();
	props.setProperty("sighanCorporaDict", basedir);
	props.setProperty("serDictionary", basedir + "/dict-chris6.ser.gz");
	props.setProperty("inputEncoding", "GBK");
	props.setProperty("sighanPostProcessing", "true");
	CRFClassifier<CoreLabel> segmenter = new CRFClassifier<CoreLabel>(props);
	segmenter.loadClassifierNoExceptions(basedir + "/ctb.gz", props);
	String sample = "";
	List<String> segmented = segmenter.segmentString(sample);
	System.out.println(segmented);
	*/
	//nlpir
	public static void seg_addwords(String[] x){
		for(String s : x){
			CLibrary.Instance.NLPIR_AddUserWord(s);
		}
	}
	public static void seg_delwords(String[] x){
		for(String s : x){
			CLibrary.Instance.NLPIR_DelUsrWord(s);
		}
	}
	//segementation --- return seg/pos
	public static Object[] get_seg(String s,String[] x){
		if(x != null)
			seg_addwords(x);
		ArrayList<String> results = new ArrayList<String>();
		ArrayList<String> resultp = new ArrayList<String>();
		StringBuilder result = new StringBuilder();
		String output = CLibrary.Instance.NLPIR_ParagraphProcess(s, 1);
		String[] split = output.split(" ");
		for(String word : split){
			String[] one = word.split("/");
			results.add(one[0]);
			if(one.length > 1 && one[1].length()>0)	//what ??
				resultp.add(one[1]);
			else
				resultp.add(NO_POS);
			result.append(one[0]+" ");
		}
		if(result.length() > 0){
			result.deleteCharAt(result.length()-1);
		}
		Object[] ret = new Object[]{results,resultp,result.toString()};
		if(x != null)
			seg_delwords(x);
		return ret;
	}
	
	/*
	//1.5 pos tagger
	public static List<String> get_pos(List<String> str){
		ArrayList<String> tags = new ArrayList<String>();
		String[] x = new String[str.size()];
		for(int i=0;i<str.size();i++)
			x[i] = (str.get(i));
		List<HasWord> sent = Sentence.toWordList(x);
		List<TaggedWord> taggedSent = tagger.tagSentence(sent);
		for (TaggedWord tw : taggedSent) {
			tags.add(tw.tag());
		}
		return tags;
	}
	
	//2.ner
	public static List<String> get_ner(String seg_str){
		if(!ner_select)
			return null;
		ArrayList<String> result = new ArrayList<String>();
		List<List<CoreLabel>> x = ner_classifier.classify(seg_str);
		for (List<CoreLabel> lcl : x) {
			for (CoreLabel cl : lcl) {
				result.add(cl.get(CoreAnnotations.AnswerAnnotation.class));
				//System.out.println(cl.toShorterString());
				//System.out.println(cl.get(CoreAnnotations.AnswerAnnotation.class));
			}
        }
		return result;
	}
	
	//3.parse --- tree/dependency
	public static Object[] get_parse(List<String> str){
		if(!parser_select)
			return null;
		String[] x = new String[str.size()];
		for(int i=0;i<str.size();i++)
			x[i] = (str.get(i));
		List<CoreLabel> rawWords = Sentence.toCoreLabelList(x);
	    Tree t = parser.apply(rawWords);
	    GrammaticalStructure gs = gsf.newGrammaticalStructure(t);
	    Collection<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
	    Object[] ret = new Object[]{t,tdl};
	    return ret;
	}
	*/
	//testing
	public static void main(String[] x){
		init(false,false);
		String sample = "我没有说我不喜欢这部电影。";
		Object[] sseg = get_seg(sample,null);
		System.out.println((List<String>)sseg[0]);
		System.out.println((List<String>)sseg[1]);
		deinit();
	}
}
