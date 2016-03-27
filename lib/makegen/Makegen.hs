{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE FlexibleInstances #-}
module Makegen
  ( module Makegen
  , mempty
  , (<>)
  ) where
import Control.Arrow
import Control.Monad
import Data.Dynamic
import Data.Maybe
import Data.Monoid
import Control.Monad.Trans.Writer.Strict
import Data.Map.Strict (Map)
import Data.Sequence (Seq)
import Data.Set (Set)
import qualified Data.Map.Strict as Map

data Makefile =
  Makefile
  { _rules :: Map String ([String], [String])
  , _macros :: Map String String
  }

unionEqMap :: (Ord k, Eq a) =>
              Map k a
           -> Map k a
           -> Either [(k, a, a)] (Map k a)
unionEqMap m1 m2
  | null conflicts = Right (m1 <> m2)
  | otherwise = Left conflicts
  where
    conflicts = do
      (k, v1) <- Map.toAscList (Map.intersection m1 m2)
      let v2 = m2 Map.! k
      guard (v1 /= v2)
      pure (k, v1, v2)

data Error
  = ERuleConflict (String, ([String], [String]), ([String], [String]))
  | EMacroConflict (String, String, String)
  deriving (Eq, Ord, Read, Show)

instance Monoid (Either [Error] Makefile) where
  mempty = Right (Makefile mempty mempty)
  mappend mx my = do
    Makefile x1 x2 <- mx
    Makefile y1 y2 <- my
    z1 <- left (ERuleConflict <$>) (unionEqMap x1 y1)
    z2 <- left (EMacroConflict <$>) (unionEqMap x2 y2)
    pure (Makefile z1 z2)

newtype VarMap
  = VarMap (Map TypeRep Dynamic)
  deriving Show

type Make a = Writer (Makefile, VarMap) a

-- | Laws are @isMempty mempty ≡ True@ and if @a ≡ t b@ and @Foldable t@ then
-- @isMempty ≡ null@.
class Monoid a => MemptyComparable a where
  isMempty :: a -> Bool

field :: (Functor f, MemptyComparable a, Typeable a) =>
         (a -> f a) -> VarMap -> f VarMap
field f m = (`set` m) <$> f (get m)
  where
    set x (VarMap m) =
      VarMap $
        if isMempty x
          then Map.delete typeRep m
          else Map.insert typeRep (toDyn x) m
      where typeRep = typeOf x
    get (VarMap m) = x
      where x = fromMaybe mempty (fromDynamic =<< Map.lookup (typeOf x) m)

data Mk m a = Mk m a
