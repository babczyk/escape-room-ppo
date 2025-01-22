using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
public class Wall : GameObject
{


    public Wall(Vector2 position, Vector2 size) : base(position, size)
    {
        gravity = 0;
    }


}
