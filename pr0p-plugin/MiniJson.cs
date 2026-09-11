// Minimal JSON parser: object -> Dictionary<string,object>, array -> List<object>,
// number -> double, string -> string, bool -> bool, null -> null.
using System;
using System.Collections.Generic;
using System.Text;

namespace Pr0pCustomModel
{
    public static class MiniJson
    {
        public static object Parse(string json)
        {
            int i = 0;
            var v = ParseValue(json, ref i);
            return v;
        }

        static void Ws(string s, ref int i)
        {
            while (i < s.Length && char.IsWhiteSpace(s[i])) i++;
        }

        static object ParseValue(string s, ref int i)
        {
            Ws(s, ref i);
            char c = s[i];
            if (c == '{') return ParseObject(s, ref i);
            if (c == '[') return ParseArray(s, ref i);
            if (c == '"') return ParseString(s, ref i);
            if (c == 't') { i += 4; return true; }
            if (c == 'f') { i += 5; return false; }
            if (c == 'n') { i += 4; return null; }
            return ParseNumber(s, ref i);
        }

        static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var d = new Dictionary<string, object>();
            i++; // {
            Ws(s, ref i);
            if (s[i] == '}') { i++; return d; }
            while (true)
            {
                Ws(s, ref i);
                string key = ParseString(s, ref i);
                Ws(s, ref i);
                i++; // :
                d[key] = ParseValue(s, ref i);
                Ws(s, ref i);
                if (s[i] == ',') { i++; continue; }
                i++; // }
                return d;
            }
        }

        static List<object> ParseArray(string s, ref int i)
        {
            var l = new List<object>();
            i++; // [
            Ws(s, ref i);
            if (s[i] == ']') { i++; return l; }
            while (true)
            {
                l.Add(ParseValue(s, ref i));
                Ws(s, ref i);
                if (s[i] == ',') { i++; continue; }
                i++; // ]
                return l;
            }
        }

        static string ParseString(string s, ref int i)
        {
            var sb = new StringBuilder();
            i++; // "
            while (s[i] != '"')
            {
                char c = s[i++];
                if (c == '\\')
                {
                    char e = s[i++];
                    switch (e)
                    {
                        case 'n': sb.Append('\n'); break;
                        case 't': sb.Append('\t'); break;
                        case 'r': sb.Append('\r'); break;
                        case 'b': sb.Append('\b'); break;
                        case 'f': sb.Append('\f'); break;
                        case 'u': sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                        default: sb.Append(e); break;
                    }
                }
                else sb.Append(c);
            }
            i++; // "
            return sb.ToString();
        }

        static double ParseNumber(string s, ref int i)
        {
            int start = i;
            while (i < s.Length && "-+0123456789.eE".IndexOf(s[i]) >= 0) i++;
            return double.Parse(s.Substring(start, i - start),
                System.Globalization.CultureInfo.InvariantCulture);
        }

        // ---- typed helpers ----
        public static Dictionary<string, object> Obj(object o)
            => o as Dictionary<string, object>;
        public static List<object> Arr(object o) => o as List<object>;
        public static double Num(object o, double def = 0)
            => o is double d ? d : def;
        public static int Int(object o, int def = 0)
            => o is double d ? (int)d : def;
        public static string Str(object o) => o as string;
        public static bool Bool(object o) => o is bool b && b;
        public static Dictionary<string, object> Get(
            Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v)
                ? v as Dictionary<string, object> : null;
        public static List<object> GetArr(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v)
                ? v as List<object> : null;
        public static int GetInt(Dictionary<string, object> d, string k, int def = 0)
            => d != null && d.TryGetValue(k, out var v) && v is double n
                ? (int)n : def;
        public static string GetStr(Dictionary<string, object> d, string k)
            => d != null && d.TryGetValue(k, out var v) ? v as string : null;
    }
}
