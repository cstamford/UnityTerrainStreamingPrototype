
public static class StringExtensions
{
    // see https://referencesource.microsoft.com/#system.servicemodel/System/ServiceModel/StringUtil.cs,9e8fa99987f83da8,references
    // this function exists to give a deterministic cross-platform hash function
    [System.Security.SecuritySafeCritical]
    public static int GetStableHashCode(this string str)
    {
        unsafe
        {
            fixed (char* src = str)
            {
                int hash1 = (5381 << 16) + 5381;
                int hash2 = hash1;

                // 32 bit machines.
                int* pint = (int*)src;
                int len = str.Length;
                while (len > 2)
                {
                    hash1 = ((hash1 << 5) + hash1 + (hash1 >> 27)) ^ pint[0];
                    hash2 = ((hash2 << 5) + hash2 + (hash2 >> 27)) ^ pint[1];
                    pint += 2;
                    len -= 4;
                }

                if (len > 0)
                {
                    hash1 = ((hash1 << 5) + hash1 + (hash1 >> 27)) ^ pint[0];
                }

                return hash1 + (hash2 * 1566083941);
            }
        }

    }
}

