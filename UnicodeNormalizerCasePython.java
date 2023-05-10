import java.io.IOException;
import java.lang.Character.UnicodeBlock;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

import static java.lang.Character.COMBINING_SPACING_MARK;
import static java.lang.Character.ENCLOSING_MARK;
import static java.lang.Character.NON_SPACING_MARK;

public class UnicodeNormalizerCasePython {
	public String evaluate(String s, int limit) {
		if (s == null) {
			return null;
		}

		if (limit == 0) {
			limit = 100;
		}

		//转小写
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < s.length();) {
			try {
				int codePoint = s.codePointAt(i);
				int newCodePoint = codePoint;
				if (Character.isUpperCase(codePoint)) {
					newCodePoint = Character.toLowerCase(codePoint);
				}
				sb.appendCodePoint(newCodePoint);
				i += Character.charCount(codePoint);
			} catch (Exception e) {
				return null;
			}
		}
		s = sb.toString();

		s = s.trim();

		s = Normalizer.normalize(s, Normalizer.Form.NFKD);

		//去掉Mn, Mc, Me这三个unicode category
		sb = new StringBuffer();
		for (int i = 0; i < s.length();) {
			try {
				int codePoint = s.codePointAt(i);
				int type = Character.getType(codePoint);
				if (type != COMBINING_SPACING_MARK && type != ENCLOSING_MARK && type != NON_SPACING_MARK) {
					sb.appendCodePoint(codePoint);
				}
				i += Character.charCount(codePoint);
			} catch (Exception e) {
				return null;
			}
		}
		s = sb.toString();

		s = s.trim();

		//以空白为分隔符分割成数组, 并以单个空格连接
		String[] part = s.split("\\s+");
		s = String.join(" ", part);

		s = s.trim();

		if (s.length() > limit) {
			return "";
		}
		return s;
	}

	public static void main(String[] args) {
		if (args.length != 1) {
			return;
		}

		UnicodeNormalizerCasePython normalizerCasePython = new UnicodeNormalizerCasePython();

		Path path = Paths.get(args[0]);

		try (Stream<String> lines = Files.lines(path)) {
			lines.forEachOrdered(line-> {
				boolean hasLabel = false;
				String label = "";
				if (line.endsWith("\t0") || line.endsWith("\t1")) {
					hasLabel = true;
					if (line.endsWith("\t0")) {
						label = "0";
					} else {
						label = "1";
					}
					line = line.substring(0, line.length() - 2);
				}
				line = normalizerCasePython.evaluate(line, 10000);
				if (hasLabel) {
					line += "\t" + label;
				}
				System.out.println(line);
			});
		} catch (IOException e) {
			//error happened
		}
	}
}
