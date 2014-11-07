package macLn.src;

public class run {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		//breast_cancer   car    primary-tumor  mushroom   lymph  kdd_ipums_la_99-small
		k_modesExp2 exp=new k_modesExp2();
		String[] expAspect={"uci","attribute","object","domain"};
		String[] uciFileNames={"soybeanSmall","cate_500_10_20_5","cate_500_20_10_5","cate_500_20_15_5",
				"cate_500_20_20_5","cate_500_20_30_5","cate_500_30_20_5",
				"soybean","car","mushroom","lymph","primary-tumor"};
		String[] path={"E:/dataset/uci/","E:/dataset/cate/attributes/","E:/dataset/cate/objects/","E:/dataset/cate/domain/"
				,"E:/dataset/cate/object2/"};
		for(int i=0;i<uciFileNames.length;++i){
			exp.main(path[0],uciFileNames[i],expAspect[0]);
		}
		String[] attrFileNames={"cate_1000_10_20_5","cate_1000_15_20_5","cate_1000_20_20_5",
								"cate_1000_30_20_5","cate_1000_50_20_5"};
		for(int i=0;i<attrFileNames.length;++i){
			exp.main(path[1],attrFileNames[i],expAspect[1]);
		}
		String[] objFileNames={"cate_1000_30_30_5","cate_2000_30_30_5","cate_3000_30_30_5",
								"cate_4000_30_30_5","cate_5000_30_30_5"};
		for(int i=0;i<objFileNames.length;++i){
			exp.main(path[2],objFileNames[i],expAspect[2]);
		}
		
		String[] domFileNames={"cate_1000_20_5_5","cate_1000_20_10_5","cate_1000_20_15_5",
								"cate_1000_20_20_5","cate_1000_20_30_5","cate_1000_20_50_5"};
		for(int i=0;i<domFileNames.length;++i){
			exp.main(path[3],domFileNames[i],expAspect[3]);
		}
		
		String[] objFileNames2={"cate_200_20_20_5","cate_400_20_20_5","cate_600_20_20_5",
				"cate_800_20_20_5","cate_1000_20_20_5"};
		for(int i=0;i<objFileNames2.length;++i){
			exp.main(path[4],objFileNames2[i],expAspect[2]);
		}
		//exp.main("soybean");
		System.out.print("ooo");
	}

}
